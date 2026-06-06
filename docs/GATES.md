# Gate Architecture: Design & Specification

## Overview

The 30B model uses a **4-gate sequential pipeline** on top of the frozen Qwen2.5-Coder-7B base. Each gate is independently trainable, composable, and interpretable.

```
Base LM (7B, frozen)
       │  hidden_state[seq_len, 4096]
       ▼
┌──────────────────────────────────────────┐
│  GATE 1: Knowledge Retrieval  (2B)     │
│  FAISS episodic memory + attention       │
│  Output: enriched_hidden + retrieval_ctx │
└──────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│  GATE 2: Symbolic Reasoning (4B)         │
│  FOL formalization + neural prover       │
│  Output: proof_trace + reasoning_hidden  │
└──────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│  GATE 3: RL Self-Correction (1B)        │
│  GRPO reward shaping + PPO              │
│  Output: value_estimate + action_logits │
└──────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│  GATE 4: Verification (3B)               │
│  Pre-streaming check                     │
│  BLOCKS bad tokens before output         │
└──────────────────────────────────────────┘
       │
       ▼
   STREAM TOKENS
```

---

## Gate 1: Knowledge Retrieval (2B params)

### Purpose
Dynamically retrieve relevant facts, code patterns, and problem-solving strategies from episodic memory before reasoning begins.

### Architecture
```
hidden_state → QueryProj → (batch, 16, 512)    # 16-head attention
              → attend to MemoryBank (100K × 8K)
              → top-K weighted sum
              → OutputProj → (batch, 4096)
              → concat [original, retrieved]
              → FusionGate → (batch, 4096)     # sigmoid-gated merge
```

### Memory Bank
- **Size**: 100,000 entries × 8,192 dims (key_dim × num_heads)
- **Content**: Learned embeddings of past successful reasoning traces
- **Update**: Online update via importance-weighted replacement
- **Retrieval**: Multi-head attention over full bank, top-32 selection

### What Gets Retrieved
- Similar past problems (same domain/difficulty)
- Relevant code patterns (from trained solutions)
- Proof strategies (from formal verification successes)
- Error patterns (what NOT to do)

### Training Signal
- Supervised: retrieve what leads to correct answers
- RL bonus: retrieval that leads to verified correct outputs

### Interfaces
```
Input:  hidden_state (batch, seq, 4096), input_tokens (batch, seq)
Output: fused_hidden (batch, 4096), metadata {
            relevance_score: float,
            top_indices: LongTensor,
            attention_scores: FloatTensor
        }
```

---

## Gate 2: Symbolic Reasoning — FOL (4B params)

### Purpose
Formalize natural language reasoning as First-Order Logic, run neural theorem-proving, and validate proof chains.

### Sub-modules

#### 2A: FOL Formalizer (1B)
Converts natural language reasoning into FOL expressions:
```
NL text → predicate extraction → argument binding → FOL formula
```
- Uses sequence-to-sequence attention over hidden state
- Maps to a vocabulary of ~10K FOL predicates
- Handles: ∀, ∃, ¬, →, ∧, ∨, equality

#### 2B: Natural Logic Inferencer (1B)
Detects entailment/contradiction between proof steps:
```
(premise_hidden, hypothesis_hidden) → entail | contradict | neutral
```
- 3-class classification with learned embeddings
- Used between consecutive proof steps for consistency

#### 2C: Neural Theorem Prover (1.5B)
Generates proof steps via learned resolution:
```
current_state + proof_history → resolution_step + rewrite_rule
```
- 8 reasoning steps per forward pass (GRUCell chain)
- 32 rewrite rules (modus ponens, resolution, unification, etc.)
- Predicts unification scores with previous 3 steps

#### 2D: Proof Validator (0.5B)
Checks if a completed proof is valid:
```
final_proof_hidden → sigmoid → validity_score
```
- Thresholds at 0.5; invalid proofs trigger retry

### Reasoning Chain
```
Step 0: Formalize premise     → FOL_0
Step 1: Apply rule R1         → FOL_1
Step 2: Apply rule R2         → FOL_2
...
Step 7: Derive conclusion     → FOL_7
        ↓
   ProofValidator → valid?
```

### Training
- Supervised: gold proof chains from formal verification datasets (LeanDojo, Coq proof trees)
- Contrastive: invalid proof steps get negative signal
- Curriculum: start with 2-step proofs, extend to 8

### Interfaces
```
Input:  hidden_state (batch, seq, 4096), reasoning_context (str)
Output: reasoning_hidden (batch, 4096), metadata {
            proof_steps: list[FOLExpression],
            proof_validity: float,
            entailment_scores: FloatTensor,
            num_steps: int
        }
```

---

## Gate 3: RL Self-Correction (1B params)

### Purpose
Learn when to self-correct after verification failures. Uses GRPO (DeepSeek-R1 style).

### Components

#### Reward Estimator
```
final_hidden → Linear(4096→2048) → GELU → Linear(2048→1024) → GELU → Linear(1024→1)
```
- Learned reward blending with hardcoded execution rewards
- Reward = 0.3 × learned + 0.7 × hardcoded

#### Value Baseline
Same architecture as reward estimator; used for advantage computation (GAE).

#### Self-Correction Predictor
```
(current_hidden, error_hidden) → concat → Linear(8192→4096) → GELU → Linear(4096→1) → sigmoid
```
- Decides: attempt correction? (threshold 0.5)
- Hard rules: always correct after 1st failure, never after 3rd

### GRPO Training Loop
```
1. Sample G responses from current policy
2. Execute code / check proofs → rewards
3. Compute advantages: reward - value_baseline
4. PPO update with clipping (ε=0.2)
5. Update value baseline via MSE
```

### PPO Loss
```
L = -min(ratio × advantage, clamp(ratio, 1-ε, 1+ε) × advantage)
```
Masked on valid token positions only.

### Interfaces
```
Input:  hidden_states (batch, seq, 4096), execution_results (list[ExecutionResult])
Output: rewards (batch,), advantages (batch,), metadata {
            should_correct: bool,
            value_estimate: float
        }
```

---

## Gate 4: Verification (3B params) — PRE-STREAMING

### Purpose
**Block bad tokens BEFORE they stream to user.** This is the last line of defense.

### Sub-modules

#### 4A: Code Executor (1B)
- Sandboxed Python execution (PyPy in subprocess)
- Neural test-case generator → generates 1-5 test cases per code block
- Runs generated code against tests → pass/fail + runtime

#### 4B: Proof Checker (1B)
- Encodes proof trace + axioms
- Predicts axiom satisfaction score
- Valid if all axioms score > 0.5

#### 4C: Consistency Checker (1B)
- Self-consistency voting: generate N responses, vote on answer
- Uses hidden state to predict consistency

### Pre-Verification Flow
```
Generated token
    │
    ▼
Is it the end of a code block? ──yes──→ Run in sandbox → passed?
    │                                        │
    no                                       yes
    │                                        ▼
    ▼                                   Stream token
Is it the end of a proof step?
    │                                        │
    yes                                      no
    ▼                                   Run proof checker
passed?                                         │
    │                                           ▼
    yes                                     Recalculate?
    ▼                                           │
Stream token ◄─────────────────────────── should_correct?
```

### Verdict Combination
```python
verdict = verdict_combiner([code_score, proof_score, consistency])
passed = verdict > 0.5
if not passed:
    trigger_self_correction()
    block_token_stream()
```

### Training
- Code: execution ground-truth (pass/fail from sandbox)
- Proofs: formal verification ground-truth
- Consistency: majority-vote agreement
- Hard negative mining: intentionally buggy code should fail

### Interfaces
```
Input:  hidden_state (batch, 4096), generated_content (str), content_type ("code"|"proof")
Output: VerificationResult {
            passed: bool,
            verdict: float,
            confidence: float,
            feedback: str | None
        }
```

---

## Parameter Budget

| Gate | Sub-module | Params |
|------|-----------|--------|
| Gate 1: Retrieval | QueryProj + Memory + Fusion | 2B |
| Gate 2: FOL | Formalizer + Inferencer + Prover + Validator | 4B |
| Gate 3: RL | Reward + Value + Correction | 1B |
| Gate 4: Verification | Executor + Checker + Consistency | 3B |
| **Total Gates** | | **10B** |
| Base (frozen) | Qwen2.5-Coder-7B | 7B |
| Projection | Hidden dim adapter | ~13B |
| **Grand Total** | | **~30B** |

---

## Sequential vs Parallel Gate Application

Gates are **sequential** (output of Gate N becomes input to Gate N+1) because:
1. Retrieval must happen before reasoning (can't reason about what you haven't retrieved)
2. FOL reasoning must happen before RL (need proof trace to evaluate)
3. RL must happen before verification (need to decide if correction is needed)
4. Verification must happen last (blocks bad tokens from streaming)

Parallel尝试 (early exit) is a future optimization: simple queries may exit early after Gate 1 if retrieval is sufficient.

---

## Missing Components (To Build)

| Component | Priority | Status |
|-----------|----------|--------|
| Sandboxed code executor (Docker/PyPy) | P0 | Not built |
| Lean/Coq proof ground-truth dataset | P0 | Not built |
| FAISS memory index population script | P1 | Not built |
| Online memory update job | P2 | Not built |
| Early-exit routing | P3 | Not built |
