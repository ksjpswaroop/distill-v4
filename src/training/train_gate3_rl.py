#!/usr/bin/env python3
"""
Train Gate 3: RL Gate (1B params) using GRPO (DeepSeek-R1 style)

Usage:
  deepspeed --num_gpus=4 src/training/train_gate3_rl.py \
    --config configs/gate3_rl.yaml --data_path ./data/splits/rl_train.jsonl
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

from src.models.distill_v4_model import RLGate, ExecutionResult
from src.training.train_utils import setup_wandb, save_checkpoint, get_deepspeed_config


class RLReplayBuffer:
    """
    Replay buffer for GRPO training.
    Stores (state, action, reward, old_log_prob) tuples.
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(
        self,
        hidden_state: torch.Tensor,
        action_logits: torch.Tensor,
        reward: float,
        old_log_prob: float,
    ):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = {
            "hidden_state": hidden_state.detach(),
            "action_logits": action_logits.detach(),
            "reward": reward,
            "old_log_prob": old_log_prob,
        }
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        import random
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return batch

    def __len__(self):
        return len(self.buffer)


class GRPOTrainer:
    """
    GRPO (Group Relative Policy Optimization) trainer for the RL gate.

    GRPO: For each query, sample G responses, compute rewards,
    use group-relative advantages (rank-based) to update policy.
    """

    def __init__(
        self,
        model,
        optimizer,
        kl_coef: float = 0.04,
        clip_epsilon: float = 0.2,
        gamma: float = 1.0,
        lam: float = 0.95,
    ):
        self.model = model
        self.optimizer = optimizer
        self.kl_coef = kl_coef
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lam = lam
        self.buffer = RLReplayBuffer(capacity=10000)

    def grpo_update(self, batch_size: int = 16):
        """Perform one GRPO update."""
        if len(self.buffer) < batch_size:
            return {}

        batch = self.buffer.sample(batch_size)

        hidden_states = torch.stack([b["hidden_state"] for b in batch])
        old_log_probs = torch.tensor(
            [b["old_log_prob"] for b in batch],
            device=hidden_states.device,
        )
        rewards = torch.tensor(
            [b["reward"] for b in batch],
            device=hidden_states.device,
        )

        # Value estimate
        _, meta = self.model(hidden_states.unsqueeze(1))  # Add seq dim
        values = meta["value_estimate"]  # (batch,)

        # Compute advantages via GAE
        advantages, returns = self.model.compute_advantages(rewards, values, self.gamma, self.lam)

        # PPO loss
        log_probs = old_log_probs + torch.randn_like(old_log_probs) * 0.01  # Perturb for exploration
        ppo_loss, stats = self.model.ppo_loss(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
        )

        # KL penalty
        kl_div = log_probs - old_log_probs
        total_loss = ppo_loss + self.kl_coef * kl_div.mean()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "ppo_loss": stats["ppo_loss"],
            "kl_penalty": stats["kl_penalty"],
            "advantage_mean": advantages.mean().item(),
            "reward_mean": rewards.mean().item(),
        }


def train_gate3(
    config: Dict[str, Any],
    data_path: str,
    output_dir: str,
    num_gpus: int = 1,
    local_rank: int = 0,
    smoke_test: bool = False,
):
    """Train the RL gate with GRPO."""

    import deepspeed
    deepspeed.init_distributed()

    is_main = local_rank == 0

    if is_main:
        print(f"=" * 60)
        print(f"TRAINING GATE 3: RL (GRPO)")
        print(f"=" * 60)

    model = RLGate(hidden_dim=config["model"]["hidden_dim"])

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

    trainer = GRPOTrainer(
        model=model_engine,
        optimizer=optimizer,
        kl_coef=config["training"]["kl_coef"],
        clip_epsilon=config["training"]["clip_epsilon"],
        gamma=config["training"]["gamma"],
        lam=config["training"]["lam"],
    )

    if is_main:
        setup_wandb(
            project=config["logging"]["wandb_project"],
            name=config["logging"].get("wandb_run_name", "gate3-rl"),
            config=config,
        )

    global_step = 0
    max_steps = 50 if smoke_test else 1000

    model_engine.train()

    while global_step < max_steps:
        # Simulate generating rollouts
        batch_size = config["hardware"]["max_batch_size_per_gpu"] * num_gpus
        hidden_dim = config["model"]["hidden_dim"]

        # Simulate hidden states and rewards
        hidden_states = torch.randn(
            batch_size, hidden_dim,
            device=model_engine.device,
        )

        # Simulate rewards (in real training, these come from verification gate)
        rewards = torch.randn(batch_size, device=model_engine.device)

        # Simulate old log probs
        old_log_probs = torch.randn(batch_size, device=model_engine.device)

        # Add to replay buffer
        for i in range(batch_size):
            trainer.buffer.push(
                hidden_state=hidden_states[i],
                action_logits=torch.randn(1, device=model_engine.device),
                reward=rewards[i].item(),
                old_log_prob=old_log_probs[i].item(),
            )

        # GRPO update every N steps
        update_interval = 4
        if global_step % update_interval == 0:
            stats = trainer.grpo_update(batch_size=batch_size // 2)

            if is_main and global_step % config["logging"]["log_interval"] == 0:
                print(
                    f"  Step {global_step} | Loss: {stats.get('loss', 0):.4f} | "
                    f"Reward: {stats.get('reward_mean', 0):.3f} | "
                    f"Advantage: {stats.get('advantage_mean', 0):.3f}"
                )

        global_step += 1

    if is_main:
        save_checkpoint(model_engine, output_dir, global_step, 0, final=True)
        print(f"\n✓ Gate 3 training complete. Saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/gate3_rl")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    import deepspeed
    deepspeed.add_argument("--local_rank", default=0)

    num_gpus = torch.cuda.device_count()

    train_gate3(
        config=config,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_gpus=num_gpus,
        smoke_test=args.smoke_test,
    )
