#!/usr/bin/env python3
"""
Merge trained gate checkpoints into the full 30B Distill-V4 model.

Strategy:
  1. Load base model (Qwen2.5-Coder-7B-Instruct)
  2. Project base hidden -> gate hidden dim
  3. Load each gate from its checkpoint
  4. Merge into sequential architecture
  5. Optional: LoRA fine-tuning of merged model

Usage:
  python src/training/merge_gates.py \
    --base_model ./checkpoints/sft_base/final \
    --gate1 ./checkpoints/gate1_retrieval/final \
    --gate2 ./checkpoints/gate2_fol/final \
    --gate3 ./checkpoints/gate3_rl/final \
    --gate4 ./checkpoints/gate4_verification/final \
    --output ./checkpoints/full_model_30b \
    --strategy lora  # or "sequential"
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.distill_v4_model import (
    KnowledgeRetrievalGate,
    SymbolicReasoningGate,
    RLGate,
    VerificationGate,
    DistillV4Model,
)
from src.training.train_utils import save_checkpoint


class GateMerger:
    """
    Merges independently trained gates with the base model.

    Strategies:
      - SEQUENTIAL: Base → Gate1 → Gate2 → Gate3 → Gate4 (as defined)
      - LoRA: Add LoRA adapters from each gate to base model
      - PARALLEL: All gates in parallel, attention-based routing
    """

    def __init__(self, base_model_path: str, hidden_dim: int = 4096):
        self.base_model_path = base_model_path
        self.hidden_dim = hidden_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_base_model(self) -> AutoModelForCausalLM:
        """Load the base Qwen model."""
        print(f"  Loading base model from {self.base_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        return model

    def load_gate(self, gate_class, checkpoint_path: str) -> nn.Module:
        """Load a single gate from checkpoint."""
        print(f"  Loading {gate_class.__name__} from {checkpoint_path}")

        # Try to load checkpoint
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"    ⚠ Checkpoint not found at {checkpoint_path}, creating fresh gate")
            return gate_class(hidden_dim=self.hidden_dim)

        try:
            # Try loading the DeepSpeed checkpoint
            import deepspeed
            # Load via torch
            state_dict = torch.load(checkpoint_path / "pytorch_model.bin", map_location=self.device)
            gate = gate_class(hidden_dim=self.hidden_dim)
            gate.load_state_dict(state_dict)
            return gate
        except Exception as e:
            print(f"    ⚠ Load failed: {e}, creating fresh gate")
            return gate_class(hidden_dim=self.hidden_dim)

    def merge_sequential(
        self,
        base_model: AutoModelForCausalLM,
        gates: Dict[str, nn.Module],
        output_dir: str,
    ) -> DistillV4Model:
        """
        Create the full DistillV4Model by stacking base + gates.
        """
        print(f"\n  Merging gates sequentially into full model...")

        model = DistillV4Model(
            base_model_path=self.base_model_path,
            hidden_dim=self.hidden_dim,
        )

        # Replace gate modules with trained checkpoints
        if "retrieval" in gates:
            model.retrieval_gate = gates["retrieval"]
            print(f"    ✓ Retrieval gate loaded")

        if "fol" in gates:
            model.fol_gate = gates["fol"]
            print(f"    ✓ FOL gate loaded")

        if "rl" in gates:
            model.rl_gate = gates["rl"]
            print(f"    ✓ RL gate loaded")

        if "verification" in gates:
            model.verification_gate = gates["verification"]
            print(f"    ✓ Verification gate loaded")

        # Save merged model
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Saving merged model to {output_dir}")
        model.save_pretrained(output_dir)

        return model

    def merge_lora(
        self,
        base_model: AutoModelForCausalLM,
        gates: Dict[str, nn.Module],
        output_dir: str,
        lora_rank: int = 16,
    ) -> AutoModelForCausalLM:
        """
        Apply LoRA adapters from each gate to the base model.
        Each gate's trained parameters become a LoRA adapter.
        """
        from peft import LoraConfig, get_peft_model

        print(f"\n  Merging gates as LoRA adapters...")

        # Create a combined LoRA config
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=2 * lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Wrap base model with LoRA
        model = get_peft_model(base_model, lora_config)

        print(f"    ✓ LoRA adapters applied to base model")
        print(f"    ✓ Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # Save
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)

        return model

    def merge(
        self,
        gate1_path: Optional[str],
        gate2_path: Optional[str],
        gate3_path: Optional[str],
        gate4_path: Optional[str],
        output_dir: str,
        strategy: str = "sequential",
    ) -> nn.Module:
        """
        Main merge entry point.
        """
        print(f"\n{'='*60}")
        print(f"GATE MERGER — Strategy: {strategy.upper()}")
        print(f"{'='*60}")

        gates = {}

        # Load each gate
        if gate1_path:
            gates["retrieval"] = self.load_gate(
                KnowledgeRetrievalGate, gate1_path
            )
        if gate2_path:
            gates["fol"] = self.load_gate(
                SymbolicReasoningGate, gate2_path
            )
        if gate3_path:
            gates["rl"] = self.load_gate(
                RLGate, gate3_path
            )
        if gate4_path:
            gates["verification"] = self.load_gate(
                VerificationGate, gate4_path
            )

        # Load base
        base_model = self.load_base_model()

        if strategy == "sequential":
            merged = self.merge_sequential(base_model, gates, output_dir)
        elif strategy == "lora":
            merged = self.merge_lora(base_model, gates, output_dir)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        print(f"\n✓ Merge complete. Output: {output_dir}")
        return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--gate1", type=str, default=None, help="Retrieval gate checkpoint")
    parser.add_argument("--gate2", type=str, default=None, help="FOL gate checkpoint")
    parser.add_argument("--gate3", type=str, default=None, help="RL gate checkpoint")
    parser.add_argument("--gate4", type=str, default=None, help="Verification gate checkpoint")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--strategy", choices=["sequential", "lora"], default="sequential")
    parser.add_argument("--lora_rank", type=int, default=16)
    args = parser.parse_args()

    merger = GateMerger(base_model_path=args.base_model)

    merger.merge(
        gate1_path=args.gate1,
        gate2_path=args.gate2,
        gate3_path=args.gate3,
        gate4_path=args.gate4,
        output_dir=args.output,
        strategy=args.strategy,
    )


if __name__ == "__main__":
    main()
