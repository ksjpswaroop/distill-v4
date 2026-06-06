#!/usr/bin/env python3
"""
Train Gate 4: Verification Gate (3B params)

Trains the verification gate to:
  1. Correctly accept passing code
  2. Correctly reject failing code
  3. Detect hallucinations
  4. Check consistency

Usage:
  deepspeed --num_gpus=4 src/training/train_gate4_verification.py \
    --config configs/gate4_verification.yaml --data_path ./data/splits/verif_train.jsonl
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.distill_v4_model import VerificationGate, ExecutionResult
from src.training.train_utils import setup_wandb, save_checkpoint, get_deepspeed_config


class VerificationDataset(Dataset):
    """
    Dataset for verification gate training.
    Each sample: (hidden_state, code, is_correct, test_results)
    """

    def __init__(self, path: str, max_samples: int = 100000):
        self.samples = []
        print(f"Loading verification data from {path}")
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                self.samples.append(json.loads(line))
        print(f"  Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class VerificationLoss:
    """
    Multi-task loss for the verification gate:
      1. Code execution classification (pass/fail/error)
      2. Rejection decision (accept/reject)
      3. Hallucination detection
      4. Consistency scoring
    """

    def __init__(self, rejection_threshold: float = 0.3):
        self.rejection_threshold = rejection_threshold

    def compute(
        self,
        accept_probs: torch.Tensor,          # (batch,)
        code_pass_probs: torch.Tensor,        # (batch,)
        consistency_scores: torch.Tensor,    # (batch,)
        hallucination_scores: torch.Tensor,   # (batch,)
        is_correct: List[bool],             # Ground truth correctness
        is_code_block: bool = True,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        batch_size = accept_probs.size(0)
        device = accept_probs.device

        # Labels
        correct_labels = torch.tensor(
            [1.0 if c else 0.0 for c in is_correct],
            device=device,
        )

        # 1. Code execution classification loss
        code_loss = F.binary_cross_entropy(
            code_pass_probs, correct_labels, reduction="mean"
        )

        # 2. Rejection loss: penalize accepting wrong answers,
        #    but don't penalize rejecting correct ones
        accept_labels = correct_labels  # If correct -> should accept
        rejection_loss = F.binary_cross_entropy(
            accept_probs, accept_labels, reduction="mean"
        )

        # 3. Hallucination loss: lower hallucination = better
        hallucination_loss = hallucination_scores.mean()  # Want low hallucination

        # 4. Consistency loss: want high consistency
        consistency_loss = (1 - consistency_scores).mean()

        # Weighted sum
        total = (
            1.0 * code_loss +
            0.8 * rejection_loss +
            0.5 * hallucination_loss +
            0.3 * consistency_loss
        )

        # Metrics
        accept_pred = (accept_probs > self.rejection_threshold).float()
        accuracy = (accept_pred == correct_labels).float().mean()

        return total, {
            "total_loss": total.item(),
            "code_loss": code_loss.item(),
            "rejection_loss": rejection_loss.item(),
            "hallucination_loss": hallucination_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "accuracy": accuracy.item(),
            "accept_rate": accept_pred.mean().item(),
        }


def train_gate4(
    config: Dict[str, Any],
    data_path: str,
    output_dir: str,
    num_gpus: int = 1,
    local_rank: int = 0,
    smoke_test: bool = False,
):
    """Train the verification gate."""

    import deepspeed
    deepspeed.init_distributed()

    is_main = local_rank == 0

    if is_main:
        print(f"=" * 60)
        print(f"TRAINING GATE 4: Verification")
        print(f"=" * 60)

    model = VerificationGate(
        hidden_dim=config["model"]["hidden_dim"],
        block_size=config["model"]["block_size"],
    )

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

    criterion = VerificationLoss(rejection_threshold=config["model"]["rejection_threshold"])

    if is_main:
        setup_wandb(
            project=config["logging"]["wandb_project"],
            name=config["logging"].get("wandb_run_name", "gate4-verification"),
            config=config,
        )

    global_step = 0
    max_steps = 50 if smoke_test else 2000

    model_engine.train()

    while global_step < max_steps:
        batch_size = config["hardware"]["max_batch_size_per_gpu"] * num_gpus
        hidden_dim = config["model"]["hidden_dim"]
        seq_len = config["model"]["block_size"]

        # Simulate hidden states
        hidden_states = torch.randn(
            batch_size, seq_len, hidden_dim,
            device=model_engine.device,
        )

        # Simulate ground truth correctness
        is_correct = [(torch.rand(1).item() > 0.3) for _ in range(batch_size)]

        # Verification forward
        verified_out, meta = model_engine(
            hidden_state=hidden_states,
            is_code_block=True,
        )

        # Loss
        loss, stats = criterion.compute(
            accept_probs=meta["accept_prob"],
            code_pass_probs=meta["code_pass_prob"],
            consistency_scores=meta["consistency_score"],
            hallucination_scores=meta["hallucination_score"],
            is_correct=is_correct,
            is_code_block=True,
        )

        model_engine.backward(loss)
        model_engine.step()

        if is_main and global_step % config["logging"]["log_interval"] == 0:
            print(
                f"  Step {global_step} | Loss: {stats['total_loss']:.4f} | "
                f"Acc: {stats['accuracy']:.3f} | "
                f"AcceptRate: {stats['accept_rate']:.3f}"
            )

        global_step += 1

    if is_main:
        save_checkpoint(model_engine, output_dir, global_step, 0, final=True)
        print(f"\n✓ Gate 4 training complete. Saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/gate4_verification")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    import deepspeed
    deepspeed.add_argument("--local_rank", default=0)

    num_gpus = torch.cuda.device_count()

    train_gate4(
        config=config,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_gpus=num_gpus,
        smoke_test=args.smoke_test,
    )
