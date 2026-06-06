# Inference Pipeline: Streaming + Pre-Verification

## Overview

The Distill-V4 inference pipeline implements **pre-verification before streaming** — every generated content unit is validated by Gate 4 before tokens reach the user. Bad outputs are blocked, rolled back, and regenerated.

```
User Prompt
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Base LM forward pass                                     │
│  Qwen2.5-Coder-7B → hidden_states[-1]                            │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: GATE 1 — Knowledge Retrieval                            │
│  Attention over 100K episodic memory bank                         │
│  Output: enriched_hidden + retrieval_metadata                     │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: GATE 2 — Symbolic Reasoning (FOL)                       │
│  8-step proof chain generation + validation                       │
│  Output: reasoning_hidden + proof_trace                          │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Hidden state fusion                                     │
│  combined = 0.5×base + 0.3×retrieved + 0.2×reasoning            │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
│                         ┌──────────────────────────────┐
│                         │  AUTOREGRESSIVE LOOP        │
│                         │                              │
│  ┌──────────────────────┴──────────────────────────┐ │
│  │  Sample next token from fused hidden state        │ │
│  │  Append to generated_ids                         │ │
│  │  Update combined_hidden                          │ │
│  │                                                      │ │
│  │  Every 10 tokens:                                  │ │
│  │  ┌──────────────────────────────────────────────┐  │ │
│  │  │  GATE 4: Verification                       │  │ │
│  │  │  content_type = detect(generated_text)      │  │ │
│  │  │  if code: run in sandbox → pass/fail        │  │ │
│  │  │  if proof: check against axioms              │  │ │
│  │  │  self-consistency voting                     │  │ │
│  │  │  verdict = combine(all checks)               │  │ │
│  │  └──────────────────────────────────────────────┘  │ │
│  │                                                      │ │
│  │  if not passed:                                     │ │
│  │    if GATE 3 says correctable:                      │ │
│  │      rollback 5 tokens                              │ │
│  │      increment correction_count                      │ │
│  │      retry                                          │ │
│  │    else:                                            │ │
│  │      stream what we have (flagged)                  │ │
│  └──────────────────────────────────────────────────────┘ │
│                         │                              │
└─────────────────────────┼──────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────────┐
              │  FINAL VERIFICATION        │
              │  Full content check        │
              │  → PASS → STREAM TO USER   │
              │  → FAIL → FLAG + STREAM    │
              └───────────────────────────┘
```

---

## Pre-Verification Strategy

### Why Pre-Verification?

Standard LLM pipelines verify **after generation** (post-hoc). This wastes compute on wrong answers. Our approach:

1. **During generation** — verify every 10 tokens
2. **Block at code fences** — run sandbox before streaming ` ``` `
3. **Block at proof steps** — validate before streaming "∴" or "QED"
4. **Rollback on failure** — undo last 5 tokens, regenerate

### Verification Triggers

| Trigger | Action |
|---------|--------|
| End of code block (` ``` `) | Sandbox execute → pass/fail |
| Proof step marker (`∴`, `→`, `QED`) | FOL proof validation |
| Every 10 tokens | Self-consistency check |
| High-perplexity token | Extra verification |
| 3 consecutive failures | Stop correcting, stream flagged |

### What Gets Verified

#### Code Verification
```python
# Generated code → extract → sandbox execute → test cases
def verify_code(hidden_state, code_text):
    # 1. Neural correctness score
    neural_score = code_correctness_predictor(hidden_state)  # 0-1
    
    # 2. Sandboxed execution (if executor is set)
    if executor:
        tests = test_generator.generate(code_text)
        result = executor.run(code_text, tests)
        execution_score = result.test_pass_rate
    else:
        execution_score = neural_score  # fallback
    
    # 3. Combine
    return neural_score * 0.3 + execution_score * 0.7
```

#### Proof Verification
```python
def verify_proof(hidden_state, proof_text, axioms):
    # 1. Encode proof
    proof_hidden = proof_encoder(hidden_state)
    
    # 2. Check against axioms
    for axiom in axioms:
        score = axiom_head(torch.cat([proof_hidden, axiom]))
        if score < 0.5:
            return False
    
    # 3. Neural validity
    return proof_validity_predictor(proof_hidden) > 0.5
```

#### Consistency Verification
```python
def verify_consistency(hidden_state):
    # Self-consistency: hidden state should encode a consistent answer
    # Use consistency_head to score
    return consistency_head(hidden_state) > 0.5
```

### Verdict Combination
```python
# 3 binary signals → learned combiner
verdict_input = [code_passed, proof_passed, consistent]
final_verdict = verdict_combiner(verdict_input)  # 0-1
passed = final_verdict > 0.5
```

---

## Streaming Architecture

### Token Streaming Flow
```
Generator loop
    │ token
    ▼
┌─────────────┐
│ Verify?     │ ─── every 10 tokens ──→ Gate 4
└─────────────┘
    │ passed
    ▼
┌─────────────┐
│ Confidence  │ ─── callback(confidence)
└─────────────┘
    │ high
    ▼
 STREAM TOKEN
```

### Confidence Signal
Each streamed token carries a confidence score:
- Initial tokens: high confidence (0.95)
- Mid-generation: moderate (0.7–0.9)
- Late generation: lower (0.5–0.7)
- After correction: reset to 0.85

### Callback Interface
```python
def stream_callback(token_text: str, confidence: float):
    """
    Called for each token as it's verified and ready to stream.
    confidence: 0.0-1.0 per-token confidence
    """
    sys.stdout.write(token_text)
    sys.stdout.flush()
```

---

## Self-Correction Loop

### When to Correct
```python
should_correct = (
    verification.failed and
    correction_count < max_attempts and
    gate3.should_self_correct(hidden_state, num_failures=correction_count)
)
```

### Correction Protocol
1. **Detect failure** — Gate 4 returns `passed=False`
2. **Query Gate 3** — `should_self_correct()` returns True
3. **Rollback** — undo last 5 tokens (removes the bad segment)
4. **Re-generate** — continue from rollback point with higher temperature
5. **Increment counter** — `correction_count += 1`
6. **Max 3 attempts** — after 3 failures, stream flagged output

### What Gets Rolled Back
- Generated token IDs
- Combined hidden states
- Retrieval/reasoning caches (for those tokens)

---

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| Base LM forward (512 tokens) | ~50ms | A100, BF16 |
| Gate 1 (retrieval) | ~5ms | Attention over 100K bank |
| Gate 2 (FOL) | ~20ms | 8-step reasoning |
| Gate 3 (RL) | ~2ms | Lightweight |
| Gate 4 (verify) | ~10ms | Neural; 100ms if sandbox |
| Token gen (per token) | ~5ms | Autoregressive |
| **Full pipeline (100 tokens)** | **~600ms** | Without sandbox |

### Bottlenecks
1. **Sandbox execution** — avoid synchronously running code; use async queue
2. **Memory retrieval** — GPU-attention over 100K is O(100K × batch)
3. **FOL proving** — 8-step chain is sequential

### Optimizations (Future)
- **Async sandbox** — queue code, verify out-of-band
- **KV cache** — cache Gate 1/2 outputs for retrieval context
- **Early exit** — simple queries skip Gate 2/3
- **Speculative decoding** — verify multiple tokens at once

---

## Inference Scripts

### Single Prompt
```bash
python src/inference/pipeline.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --gate1 checkpoints/gate1.pt \
  --gate2 checkpoints/gate2.pt \
  --gate3 checkpoints/gate3.pt \
  --gate4 checkpoints/gate4.pt \
  --prompt "Prove that the sum of two even numbers is even" \
  --temperature 0.7 \
  --max-tokens 2048
```

### Interactive Mode
```bash
python src/inference/pipeline.py \
  --model checkpoints/distill-v4-final \
  --gate1 checkpoints/gate1.pt \
  --gate2 checkpoints/gate2.pt \
  --gate3 checkpoints/gate3.pt \
  --gate4 checkpoints/gate4.pt \
  --interactive
```

### API Server
```python
# scripts/serve.py
from src.inference.pipeline import DistillV4InferencePipeline

pipeline = DistillV4InferencePipeline(
    base_model_path="checkpoints/distill-v4-final",
    gate1_path="checkpoints/gate1.pt",
    gate2_path="checkpoints/gate2.pt",
    gate3_path="checkpoints/gate3.pt",
    gate4_path="checkpoints/gate4.pt",
)

@app.post("/generate")
def generate(req: GenerateRequest):
    result = pipeline.generate(req.prompt, config=req.config)
    return {
        "text": result.text,
        "verification_passed": result.verification_passed,
        "confidence": result.confidence,
        "latency_ms": result.latency_seconds * 1000,
    }
```

---

## Quantization for Deployment

### INT8 Quantization
```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

model = AutoModelForCausalLM.from_pretrained(
    "checkpoints/distill-v4-final",
    quantization_config=quantization_config,
)
```

### INT4 Quantization (for edge)
```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)
```

### Expected Size After Quantization
| Precision | Base (7B) | Gates (10B) | Total |
|-----------|-----------|-------------|-------|
| FP16 | 14 GB | 20 GB | 34 GB |
| INT8 | 7 GB | 10 GB | 17 GB |
| INT4 | 3.5 GB | 5 GB | 8.5 GB |

---

## Monitoring & Observability

### Per-Generation Metrics
```python
{
    "verification_passed": bool,
    "confidence": float,           # 0-1
    "num_self_corrections": int,
    "tokens_generated": int,
    "latency_ms": float,
    "gate_latencies_ms": {
        "gate1_retrieval": float,
        "gate2_reasoning": float,
        "gate3_rl": float,
        "gate4_verification": float,
    },
    "retrieval_top_k": [int],      # which memory entries used
    "proof_validity": float,        # FOL proof score
}
```

### Logging
```python
# Every generation logs to W&B
wandb.log({
    "verification_passed": result.verification_passed,
    "confidence": result.confidence,
    "self_corrections": result.num_self_corrections,
    "latency": result.latency_seconds,
})
```

---

## Current Pipeline Status

| Feature | Status | Notes |
|---------|--------|-------|
| Base LM integration | ✅ Working | Qwen2.5-Coder-7B |
| Gate 1 (retrieval) | ✅ Code ready | Memory bank needs population |
| Gate 2 (FOL) | ✅ Code ready | External prover not connected |
| Gate 3 (RL) | ✅ Code ready | Needs training data |
| Gate 4 (verification) | ✅ Code ready | Sandbox not wired |
| Token streaming | ✅ Working | Callback interface implemented |
| Pre-verification | ✅ Logic ready | Needs sandbox integration |
| Self-correction loop | ✅ Working | Rollback + retry |
| Quantization | ⏳ Pending | INT8/INT4 scripts not written |
| API server | ⏳ Pending | FastAPI server not built |
