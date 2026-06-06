#!/usr/bin/env python3
"""
SFT Base Model Training — Phase 2

Fine-tune Qwen2.5-Coder-7B on distillation data using DeepSpeed ZeRO-2.
This is the foundation before gate training.

Usage:
  # Smoke test (1 GPU)
  deepspeed --num_gpus=1 src/training/train_sft_base.py \
    --config configs/sft_base.yaml --data_path ./data/splits/train.jsonl --smoke_test

  # Full training (8 GPUs)
  deepspeed --num_gpus=8 src/training/train_sft_base.py \
    --config configs/sft_base.yaml \
    --data_path ./data/splits/train.jsonl \
    --output_dir ./checkpoints/sft_base
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training.train_utils import setup_wandb, save_checkpoint, get_deepspeed_config, print_gpu_memory


class SFTDataset(Dataset):
    """SFT dataset: (question, answer) pairs."""

    def __init__(self, path: str, tokenizer, max_length: int = 8192):
        self.samples = []
        self.tokenizer = tokenizer
        print(f"Loading SFT data from {path}")
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i >= 200_000:
                    break
                sample = json.loads(line)
                self.samples.append(sample)
        print(f"  Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        question = sample.get("question", "")
        answer = sample.get("reference_answer", "")

        # Format: Q: {question}\n\nA: {answer}
        full_text = f"Q: {question}\n\nA: {answer}"

        enc = self.tokenizer(
            full_text,
            max_length=8192,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": enc["input_ids"].squeeze(0).clone(),
        }


def train_sft(
    config: Dict[str, Any],
    data_path: str,
    output_dir: str,
    num_gpus: int = 1,
    local_rank: int = 0,
    smoke_test: bool = False,
):
    """Fine-tune base model via SFT."""

    deepspeed.init_distributed()
    is_main = local_rank == 0

    if is_main:
        print(f"=" * 60)
        print(f"SFT BASE MODEL TRAINING")
        print(f"=" * 60)
        print(f"  Data: {data_path}")
        print(f"  Output: {output_dir}")
        print(f"  GPUs: {num_gpus}")
        print_gpu_memory()

    # Load base model
    model_name = config["model"].get("name", "Qwen/Qwen2.5-Coder-7B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    dataset = SFTDataset(data_path, tokenizer)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # DeepSpeed config
    ds_config = get_deepspeed_config(config["deepspeed"]["config_path"])
    ds_config["train_batch_size"] = config["data"]["batch_size"] * num_gpus

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
    )

    if is_main:
        setup_wandb(
            project=config["logging"]["wandb_project"],
            name=config["logging"].get("wandb_run_name", "sft-base"),
            config=config,
        )

    global_step = 0
    epochs = config["training"]["num_epochs"]
    max_steps = 20 if smoke_test else -1

    model_engine.train()

    for epoch in range(epochs):
        dataloader = DataLoader(
            dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=True,
            num_workers=config["data"].get("num_workers", 4),
            pin_memory=True,
        )

        for batch in dataloader:
            if max_steps > 0 and global_step >= max_steps:
                break

            input_ids = batch["input_ids"].to(model_engine.device)
            attention_mask = batch["attention_mask"].to(model_engine.device)
            labels = batch["labels"].to(model_engine.device)

            # Forward
            outputs = model_engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            # Backward
            model_engine.backward(loss)
            model_engine.step()

            if is_main and global_step % config["logging"]["log_interval"] == 0:
                print(
                    f"  Step {global_step} | Loss: {loss.item():.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )

            global_step += 1

        if is_main:
            print(f"\n  Epoch {epoch} complete")

        if is_main and global_step % config["logging"]["save_interval"] == 0:
            save_checkpoint(model_engine, output_dir, global_step, epoch)

    if is_main:
        save_checkpoint(model_engine, output_dir, global_step, epoch, final=True)
        print(f"\n✓ SFT training complete. Checkpoints: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/sft_base")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    deepspeed.add_argument("--local_rank", default=0)

    num_gpus = torch.cuda.device_count()

    train_sft(
        config=config,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_gpus=num_gpus,
        smoke_test=args.smoke_test,
    )
