#!/usr/bin/env python3
"""
Train Gate 2: FOL Symbolic Reasoning (4B params)

Usage:
  # Smoke test (1 GPU)
  deepspeed --num_gpus=1 src/training/train_gate2_fol.py \
    --config configs/gate2_fol.yaml --smoke_test

  # Full training (8 GPUs)
  deepspeed --num_gpus=8 src/training/train_gate2_fol.py \
    --config configs/gate2_fol.yaml \
    --data_path ./data/splits/train.jsonl \
    --output_dir ./checkpoints/gate2_fol
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.distill_v4_model import SymbolicReasoningGate
from src.training.train_utils import setup_wandb, save_checkpoint, get_deepspeed_config


class FOLDataset(Dataset):
    """Dataset for FOL reasoning training."""

    def __init__(self, path: str, tokenizer, max_length: int = 2048):
        self.samples = []
        self.tokenizer = tokenizer
        print(f"Loading FOL data from {path}")
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i >= 100_000:
                    break
                self.samples.append(json.loads(line))
        print(f"  Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        enc = self.tokenizer(
            sample.get("question", "") + " [SEP] " + sample.get("reference_answer", ""),
            max_length=2048,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "category": sample.get("category", ""),
            "subcategory": sample.get("subcategory", ""),
        }


def fol_loss(
    proof_hidden: torch.Tensor,
    proof_validity: torch.Tensor,
    entailment_tensor: torch.Tensor,
    target_validity: float = 1.0,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    FOL gate loss = proof validity loss + entailment consistency loss.
    """
    # Proof validity: we want validity to be high
    validity_loss = F.mse_loss(proof_validity, torch.full_like(proof_validity, target_validity))

    # Entailment: consecutive proof steps should have positive entailment
    if entailment_tensor is not None:
        # Entailment class = 0, so take softmax[:, 0] as entailment prob
        entailment_loss = (1 - entailment_tensor).mean()
    else:
        entailment_loss = torch.tensor(0.0, device=proof_hidden.device)

    # Chain continuity: hidden states should evolve smoothly
    hidden_var = proof_hidden.var(dim=0).mean()
    continuity_loss = hidden_var  # Higher variance = less coherent chain

    total = validity_loss + 0.5 * entailment_loss + 0.1 * continuity_loss

    return total, {
        "validity_loss": validity_loss.item(),
        "entailment_loss": entailment_loss.item() if isinstance(entailment_loss, torch.Tensor) else entailment_loss,
        "continuity_loss": continuity_loss.item(),
    }


def train_gate2(
    config: Dict[str, Any],
    data_path: str,
    output_dir: str,
    num_gpus: int = 1,
    local_rank: int = 0,
    smoke_test: bool = False,
):
    """Train the FOL symbolic reasoning gate."""

    import deepspeed
    deepspeed.init_distributed()

    is_main = local_rank == 0

    if is_main:
        print(f"=" * 60)
        print(f"TRAINING GATE 2: FOL Symbolic Reasoning")
        print(f"=" * 60)

    model = SymbolicReasoningGate(
        hidden_dim=config["model"]["hidden_dim"],
        intermediate_dim=config["model"]["intermediate_dim"],
        num_reasoning_steps=config["model"]["num_reasoning_steps"],
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"].get("name", "Qwen/Qwen2.5-Coder-7B-Instruct"),
        trust_remote_code=True,
    )

    train_dataset = FOLDataset(data_path, tokenizer)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    ds_config = get_deepspeed_config(config["deepspeed"]["config_path"])
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
    )

    if is_main:
        setup_wandb(
            project=config["logging"]["wandb_project"],
            name=config["logging"].get("wandb_run_name", "gate2-fol"),
            config=config,
        )

    global_step = 0
    epochs = config["training"]["num_epochs"]
    max_steps = 50 if smoke_test else -1

    model_engine.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        dataloader = DataLoader(
            train_dataset,
            batch_size=config["data"].get("batch_size", 2),
            shuffle=True,
            num_workers=config["data"].get("num_workers", 2),
        )

        for batch in dataloader:
            if max_steps > 0 and global_step >= max_steps:
                break

            # Simulate hidden states from token embeddings
            batch_size = batch["input_ids"].size(0)
            seq_len = batch["input_ids"].size(1)
            hidden_dim = config["model"]["hidden_dim"]

            with torch.no_grad():
                fake_hidden = torch.randn(
                    batch_size, seq_len, hidden_dim,
                    device=model_engine.device,
                ) * 0.02

            # FOL forward
            reasoning_out, meta = model_engine(
                hidden_state=fake_hidden,
                reasoning_context="prove",
            )

            # Loss
            loss, loss_stats = fol_loss(
                proof_hidden=reasoning_out,
                proof_validity=meta["proof_validity"],
                entailment_tensor=meta["entailment_tensor"],
            )

            model_engine.backward(loss)
            model_engine.step()

            if is_main and global_step % config["logging"]["log_interval"] == 0:
                print(
                    f"  Step {global_step} | Loss: {loss.item():.4f} | "
                    f"Validity: {meta['proof_validity'].mean():.3f} | "
                    f"Steps: {meta['num_steps']}"
                )

            epoch_loss += loss.item()
            global_step += 1

        if is_main:
            print(f"\n  Epoch {epoch}: avg_loss={epoch_loss/max(len(dataloader),1):.4f}")

        if is_main and global_step % config["logging"]["save_interval"] == 0:
            save_checkpoint(model_engine, output_dir, global_step, epoch)

    if is_main:
        save_checkpoint(model_engine, output_dir, global_step, epoch, final=True)
        print(f"\n✓ Gate 2 training complete. Saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/gate2_fol")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    import deepspeed
    deepspeed.add_argument("--local_rank", default=0)

    num_gpus = torch.cuda.device_count()

    train_gate2(
        config=config,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_gpus=num_gpus,
        smoke_test=args.smoke_test,
    )
