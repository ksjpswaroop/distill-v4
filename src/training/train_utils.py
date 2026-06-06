"""
Training utilities shared across all gate trainers.
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional


def setup_wandb(
    project: str,
    name: Optional[str] = None,
    config: Optional[Dict] = None,
    entity: Optional[str] = None,
):
    """
    Initialize Weights & Biases logging.
    """
    try:
        import wandb
        wandb.init(
            project=project,
            name=name,
            config=config,
            entity=entity,
        )
        print(f"  [WandB] Initialized: project={project}, name={name}")
    except ImportError:
        print(f"  [WandB] Not installed, skipping")
    except Exception as e:
        print(f"  [WandB] Init failed: {e}")


def log_metrics(
    metrics: Dict[str, float],
    step: int,
    prefix: str = "train",
):
    """Log metrics to WandB."""
    try:
        import wandb
        wandb.log({f"{prefix}/{k}": v for k, v in metrics.items()}, step=step)
    except Exception:
        pass


def save_checkpoint(
    model_engine,
    output_dir: str,
    step: int,
    epoch: int,
    final: bool = False,
):
    """
    Save a DeepSpeed checkpoint.
    """
    import deepspeed

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save using DeepSpeed save
    save_path = output_dir / f"checkpoint-step{step}"
    model_engine.save_checkpoint(save_path)

    # Save metadata
    meta = {
        "step": step,
        "epoch": epoch,
        "final": final,
    }
    with open(save_path / "meta.json", "w") as f:
        json.dump(meta, f)

    print(f"  [Checkpoint] Saved to {save_path}")


def load_checkpoint(
    model_engine,
    checkpoint_path: str,
):
    """Load a DeepSpeed checkpoint."""
    import deepspeed
    model_engine.load_checkpoint(checkpoint_path)


def get_deepspeed_config(config_path: str) -> Dict[str, Any]:
    """Load DeepSpeed JSON config."""
    import json
    with open(config_path) as f:
        return json.load(f)


def get_adam_optimizer(model, lr: float, weight_decay: float):
    """Create AdamW optimizer."""
    no_decay = ["bias", "LayerNorm", "layernorm", "norm"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)


def get_cosine_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int):
    """Cosine learning rate scheduler."""
    from torch.optim.lr_scheduler import CosineAnnealingLR
    return CosineAnnealingLR(optimizer, T_max=num_training_steps)


class CountableTimer:
    """Simple timer for profiling."""

    def __init__(self, name: str = ""):
        self.name = name
        self.times = []
        self._start = None
        import time
        self._time = time

    def __enter__(self):
        self._start = self._time.time()
        return self

    def __exit__(self, *args):
        elapsed = self._time.time() - self._start
        self.times.append(elapsed)
        if self.name:
            print(f"  [{self.name}] {elapsed:.3f}s")


def format_num(n: int) -> str:
    """Format large numbers nicely."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
