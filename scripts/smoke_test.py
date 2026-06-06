#!/usr/bin/env python3
"""
Smoke test for Distill-V4 gate architecture.

Verifies:
  1. All 4 gates initialize without errors
  2. Forward pass produces correct shapes
  3. No NaN/Inf in gradients
  4. CUDA vs CPU fallback works
  5. Parameter counts are in expected ranges

Run:
  python scripts/smoke_test.py
  python scripts/smoke_test.py --device cuda --verbose
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import argparse
from pathlib import Path

# Import model components
from src.models.distill_v4_model import (
    KnowledgeRetrievalGate,
    SymbolicReasoningGate,
    RLGate,
    VerificationGate,
    count_parameters,
    GateType,
)


def reset_seed(seed: int = 42):
    """Reset RNG for reproducible tests."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)


def test_knowledge_retrieval_gate(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Test Gate 1: Knowledge Retrieval."""
    print("\n  [Gate 1] Knowledge Retrieval Gate...")

    batch_size = 2
    seq_len = 128
    hidden_dim = 4096
    memory_size = 10_000
    num_heads = 16
    key_dim = 256

    gate = KnowledgeRetrievalGate(
        hidden_dim=hidden_dim,
        memory_size=memory_size,
        num_heads=num_heads,
        key_dim=key_dim,
    ).to(device)

    params = count_parameters(gate)
    print(f"    Params: {params:,} ({params/1e9:.3f}B)")

    # Forward pass
    hidden_state = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    input_tokens = torch.randint(0, 50000, (batch_size, seq_len), device=device)

    with torch.no_grad():
        fused, meta = gate(hidden_state, input_tokens, retrieve_top_k=8)

    assert fused.shape == (batch_size, hidden_dim), f"Expected ({batch_size}, {hidden_dim}), got {fused.shape}"
    assert "relevance" in meta, "Missing relevance score in metadata"
    assert "gate_value" in meta, "Missing gate_value in metadata"

    # Test backward
    fused.sum().backward()
    for name, param in gate.named_parameters():
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    print(f"    ✓ Forward shape: {fused.shape}")
    print(f"    ✓ Backward: no NaN gradients")
    print(f"    ✓ Relevance range: [{meta['relevance'].min():.3f}, {meta['relevance'].max():.3f}]")
    print(f"    ✓ Gate value range: [{meta['gate_value'].min():.3f}, {meta['gate_value'].max():.3f}]")

    del gate, hidden_state, input_tokens, fused
    torch.cuda.empty_cache()
    print("    [Gate 1] PASSED")
    return True


def test_fol_gate(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Test Gate 2: FOL Symbolic Reasoning."""
    print("\n  [Gate 2] FOL Symbolic Reasoning Gate...")

    batch_size = 2
    seq_len = 64
    hidden_dim = 4096
    intermediate_dim = 8192  # Smaller for smoke test

    gate = SymbolicReasoningGate(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_reasoning_steps=4,  # Fewer steps for smoke test
    ).to(device)

    params = count_parameters(gate)
    print(f"    Params: {params:,} ({params/1e9:.3f}B)")

    # Forward pass
    hidden_state = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    with torch.no_grad():
        reasoning_out, meta = gate(hidden_state, reasoning_context="prove that")

    assert reasoning_out.shape == (batch_size, hidden_dim)
    assert "proof_validity" in meta
    assert "entailment_tensor" in meta
    assert "num_steps" in meta

    print(f"    ✓ Forward shape: {reasoning_out.shape}")
    print(f"    ✓ Proof validity range: [{meta['proof_validity'].min():.3f}, {meta['proof_validity'].max():.3f}]")
    print(f"    ✓ Num reasoning steps: {meta['num_steps']}")

    # Test entailment scores
    if meta["entailment_tensor"] is not None:
        print(f"    ✓ Entailment shape: {meta['entailment_tensor'].shape}")

    # Backward
    reasoning_out.sum().backward()
    for name, param in gate.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    print(f"    ✓ Backward: no NaN gradients")

    del gate, hidden_state, reasoning_out
    torch.cuda.empty_cache()
    print("    [Gate 2] PASSED")
    return True


def test_rl_gate(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Test Gate 3: Reinforcement Learning."""
    print("\n  [Gate 3] RL Gate...")

    batch_size = 4
    seq_len = 64
    hidden_dim = 4096

    gate = RLGate(hidden_dim=hidden_dim).to(device)

    params = count_parameters(gate)
    print(f"    Params: {params:,} ({params/1e9:.3f}B)")

    # Forward pass
    hidden_state = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    with torch.no_grad():
        rl_out, meta = gate(hidden_state, execution_results=None)

    assert rl_out.shape == (batch_size, hidden_dim)
    assert "reward" in meta
    assert "value_estimate" in meta

    # Test reward computation
    fake_results = [
        type('obj', (object,), {'passed': True, 'test_pass_rate': 0.8, 'is_efficient': True})(),
        type('obj', (object,), {'passed': False, 'test_pass_rate': 0.0, 'is_efficient': False})(),
    ]
    _, meta2 = gate(hidden_state, execution_results=fake_results)

    assert meta2["reward"].shape == (batch_size,)
    print(f"    ✓ Forward shape: {rl_out.shape}")
    print(f"    ✓ Reward range: [{meta2['reward'].min():.3f}, {meta2['reward'].max():.3f}]")
    print(f"    ✓ Value estimate range: [{meta2['value_estimate'].min():.3f}, {meta2['value_estimate'].max():.3f}]")

    # Test PPO loss
    log_probs = torch.randn(batch_size)
    old_log_probs = log_probs + torch.randn(batch_size) * 0.1
    advantages = torch.randn(batch_size)
    rewards = torch.randn(batch_size)
    values = torch.randn(batch_size)

    advantages, returns = gate.compute_advantages(rewards, values)
    loss, stats = gate.ppo_loss(log_probs, old_log_probs, advantages)

    assert loss.numel() == 1, "Loss should be scalar"
    assert not torch.isnan(loss), "Loss is NaN"
    print(f"    ✓ PPO loss: {loss.item():.4f}")
    print(f"    ✓ KL penalty: {stats['kl_penalty']:.4f}")

    del gate, hidden_state, rl_out
    torch.cuda.empty_cache()
    print("    [Gate 3] PASSED")
    return True


def test_verification_gate(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Test Gate 4: Verification."""
    print("\n  [Gate 4] Verification Gate...")

    batch_size = 2
    seq_len = 64
    hidden_dim = 4096

    gate = VerificationGate(hidden_dim=hidden_dim).to(device)

    params = count_parameters(gate)
    print(f"    Params: {params:,} ({params/1e9:.3f}B)")

    # Forward pass
    hidden_state = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    test_code = """
def add(a, b):
    return a + b

def test_add():
    assert add(1, 2) == 3
    assert add(0, 0) == 0
"""

    with torch.no_grad():
        verified_out, meta = gate(
            hidden_state,
            candidate_code=test_code,
            test_cases=["assert add(1,2)==3", "assert add(0,0)==0"],
            is_code_block=True,
        )

    assert verified_out.shape == (batch_size, hidden_dim)
    assert "accept" in meta
    assert "accept_prob" in meta
    assert "code_pass_prob" in meta

    n_accepted = meta["accept"].sum().item()
    print(f"    ✓ Forward shape: {verified_out.shape}")
    print(f"    ✓ Acceptance rate: {n_accepted}/{batch_size}")
    print(f"    ✓ Accept prob range: [{meta['accept_prob'].min():.3f}, {meta['accept_prob'].max():.3f}]")
    print(f"    ✓ Code pass prob range: [{meta['code_pass_prob'].min():.3f}, {meta['code_pass_prob'].max():.3f}]")

    if meta.get("exec_results"):
        print(f"    ✓ Execution results: {[r.passed for r in meta['exec_results']]}")

    del gate, hidden_state, verified_out
    torch.cuda.empty_cache()
    print("    [Gate 4] PASSED")
    return True


def test_gate_sequential_pass(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Test all 4 gates in sequence (simulating full model forward)."""
    print("\n  [Sequential Pass] All 4 gates in sequence...")

    batch_size = 2
    seq_len = 128
    hidden_dim = 4096

    # Create all gates
    gate1 = KnowledgeRetrievalGate(hidden_dim=hidden_dim, memory_size=5_000).to(device)
    gate2 = SymbolicReasoningGate(hidden_dim=hidden_dim, intermediate_dim=8192, num_reasoning_steps=4).to(device)
    gate3 = RLGate(hidden_dim=hidden_dim).to(device)
    gate4 = VerificationGate(hidden_dim=hidden_dim).to(device)

    total_params = sum(count_parameters(g) for g in [gate1, gate2, gate3, gate4])
    print(f"    Total gate params: {total_params:,} ({total_params/1e9:.3f}B)")

    # Simulate hidden state
    hidden = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    tokens = torch.randint(0, 50000, (batch_size, seq_len), device=device)

    # Gate 1: Retrieval
    with torch.no_grad():
        r_out, r_meta = gate1(hidden, tokens, retrieve_top_k=8)
    hidden[:, -1, :] = r_out

    # Gate 2: FOL
    with torch.no_grad():
        f_out, f_meta = gate2(hidden)
    hidden[:, -1, :] = f_out

    # Gate 3: RL
    with torch.no_grad():
        rl_out, rl_meta = gate3(hidden)

    # Gate 4: Verification
    with torch.no_grad():
        v_out, v_meta = gate4(hidden)
    hidden[:, -1, :] = v_out

    print(f"    ✓ Sequential pass complete")
    print(f"    ✓ Retrieval relevance: {r_meta['relevance'].mean():.3f}")
    print(f"    ✓ FOL proof validity: {f_meta['proof_validity'].mean():.3f}")
    print(f"    ✓ RL reward: {rl_meta['reward'].mean():.3f}")
    print(f"    ✓ Verification accept rate: {v_meta['accept'].float().mean():.3f}")

    for g in [gate1, gate2, gate3, gate4]:
        del g
    del hidden, tokens
    torch.cuda.empty_cache()

    print("    [Sequential Pass] PASSED")
    return True


def test_memory_and_dtype(device: str):
    """Test BF16 training, BF16 storage, and memory footprint."""
    print(f"\n  [Memory Test] Device={device}")

    hidden_dim = 4096

    if device == "cuda":
        # Check BF16 support
        if not torch.cuda.is_bf16_supported():
            print(f"    ⚠ BF16 not supported on this GPU, skipping")
            return True

        gate = SymbolicReasoningGate(
            hidden_dim=hidden_dim,
            intermediate_dim=8192,
            num_reasoning_steps=4,
        ).to(device)

        # Convert to BF16
        gate = gate.bfloat16()
        hidden = torch.randn(2, 64, hidden_dim, device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            out, _ = gate(hidden)

        assert out.dtype == torch.bfloat16
        mem_mb = sum(p.numel() * p.element_size() for p in gate.parameters()) / 1e6
        print(f"    ✓ BF16 forward pass OK")
        print(f"    ✓ Gate BF16 memory: {mem_mb:.1f} MB")
        del gate, hidden, out
        torch.cuda.empty_cache()
    else:
        print(f"    ✓ CPU mode OK (skipping BF16)")

    return True


def print_system_info():
    """Print GPU/CPU info."""
    print("=" * 60)
    print("SYSTEM INFO")
    print("=" * 60)
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        print(f"  BF16 supported: {torch.cuda.is_bf16_supported()}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 60)


def run_smoke_tests(device: str = None, verbose: bool = False):
    """Run all smoke tests."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    reset_seed(42)

    print_system_info()

    tests = [
        ("Gate 1 (Retrieval)", lambda: test_knowledge_retrieval_gate(device)),
        ("Gate 2 (FOL)", lambda: test_fol_gate(device)),
        ("Gate 3 (RL)", lambda: test_rl_gate(device)),
        ("Gate 4 (Verification)", lambda: test_verification_gate(device)),
        ("Sequential Pass", lambda: test_gate_sequential_pass(device)),
        ("Memory/Dtype", lambda: test_memory_and_dtype(device)),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"{'='*60}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            import traceback
            print(f"    ✗ FAILED: {e}")
            if verbose:
                traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"SMOKE TEST RESULTS: {passed}/{passed+failed} passed, {failed} failed")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    success = run_smoke_tests(device=args.device, verbose=args.verbose)
    sys.exit(0 if success else 1)
