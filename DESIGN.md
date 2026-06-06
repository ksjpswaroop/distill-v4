# Distill-V4 Design Document

## All Thoughts, Decisions, and Design Rationale

---

## 1. Project Vision

**What are we building?**
A 30B parameter reasoning model that distills DeepSeek-V4's English-language coding, problem-solving, and reasoning capabilities into a specialized student model with four trainable inference-time gating mechanisms.

**Why this matters:**
- DeepSeek-V4 is a large, expensive teacher model
- We want a faster, specialized student that retains the coding/reasoning capabilities
- The 4-gate architecture allows modular improvement and interpretability
- Focus on English-only simplifies the distillation and improves quality

**End goal:**
A production-deployable 30B model that can:
1. Write high-quality code (HumanEval 85%+)
2. Prove mathematical theorems formally
3. Self-correct its own errors via RL
4. Verify outputs before streaming

---

## 2. Seed Model Selection

### Why Qwen2.5-Coder-7B-Instruct?

| Criteria | Qwen2.5-Coder-7B | DeepSeek-Coder-6.7B | CodeLlama-7B |
|----------|-----------------|---------------------|--------------|
| HumanEval | **88.4** | 78.2 | 53.8 |
| MBPP | **82.1** | 75.8 | 58.2 |
| Context | **128K** | 16K | 16K |
| Python-specialized | Yes | Yes | Somewhat |

**Decision: Qwen2.5-Coder-7B-Instruct as PRIMARY seed**

Rationale:
1. Highest coding benchmarks at 7B size
2. 128K context essential for complex multi-file reasoning
3. Already instruction-tuned (saves SFT warmup)
4. Qwen architecture is well-optimized for inference
5. DeepSeek family compatibility helps with distillation

**SECONDARY:** DeepSeek-Coder-6.7B-Instruct (for ablation studies)

### Why not larger models?
- Qwen2.5-14B-Coder would be better quality but pushes us to 40B+ expanded
- We want fast iteration during development
- 7B → 30B expansion is the sweet spot for our compute budget

### Expansion Strategy: 7B → 30B
1. **Method A (Recommended):** Add 8 transformer layers + expand embeddings
   - +5B params from new layers
   - +3B params from embedding expansion
   - Knowledge distill to fill new capacity
2. **Method B:** LoRA adapters (faster iteration, slightly lower quality)
3. **Method C:** Parallel domains (fast path + reasoning path)

---

## 3. The Four Gates Architecture

### Why gates?

Traditional LLM: `input → LM → output`

Our architecture: `input → LM → Gate1 → Gate2 → Gate3 → Gate4 → output`

Each gate adds a trainable capability without retraining the entire model.
Gates can be improved independently.
Verification (Gate 4) prevents bad outputs from streaming.

### Gate 1: Knowledge Retrieval (2B params)

**Problem it solves:**
- LMs have limited context window
- Can't memorize all code patterns, algorithms, theorems
- Need dynamic retrieval from episodic memory

**Architecture:**
```
Query projection → Attention over memory bank → Retrieved knowledge
                                                              ↓
                                    Fusion gate: g * retrieved + (1-g) * hidden
```

**What it retrieves:**
- Similar problems solved previously
- Relevant code patterns from training
- Factual knowledge from knowledge bases
- Cross-references to related concepts

**Memory design:**
- Learned episodic memory bank (100K entries × 4096 dims)
- Attention-based retrieval (not exact match)
- Online learning to add new memories
- Importance-weighted replacement policy

### Gate 2: Symbolic Reasoning (4B params)

**Problem it solves:**
- Neural networks struggle with formal logic
- Need rigorous mathematical proofs
- Want verifiable reasoning chains

**Sub-modules:**

1. **FOL Formalizer (1B):** Converts natural language to First-Order Logic
   - Input: "All humans are mortal, Socrates is human"
   - Output: ∀x (Human(x) → Mortal(x)), Human(Socrates)

2. **Natural Logic Inferencer (1B):** Detects entailment/contradiction
   - Forward entailment (general → specific)
   - Backward entailment (specific → general)
   - Negation (contradiction detection)

3. **Symbolic Reasoner (1.5B):** Neural theorem prover interface
   - Resolution-based proving
   - Natural deduction
   - Rewriting systems
   - Invariant checking

4. **Proof Validator (0.5B):** Validates proof correctness
   - FOL syntax validation
   - Logical consistency checking
   - Counterexample detection

**Integration with LM:**
- Reasoning trace is injected back into hidden state
- Final hidden = blend(neural, symbolic)
- Proof steps available for verification gate

### Gate 3: Reinforcement Learning (1B params)

**Problem it solves:**
- Static SFT can't learn from consequences
- Model needs to learn from code execution results
- Self-correction requires feedback loops

**Architecture:**

1. **Reward Estimator:** Learns to predict reward from hidden state
   - Blends learned reward with hardcoded metrics
   - 0.3 * learned + 0.4 * correctness + 0.2 * code + 0.1 * efficiency

2. **Value Baseline:** Computes advantage for PPO
   - GAE (Generalized Advantage Estimation)
   - Reduces variance in policy updates

3. **Self-Correction Predictor:** Decides when to correct
   - Takes (hidden_state, error_context)
   - Returns probability that correction will help
   - Rules: Always correct after 1st failure, never after 3rd

**Training: GRPO (DeepSeek-style)**
- Generate G=8 responses per question
- Compute rewards, normalize within group
- Update policy using group-relative advantages
- More stable than vanilla PPO for LMs

**Self-Correction Loop:**
```
If verification fails AND corrections < max_attempts:
    Rollback N tokens
    Inject error context into hidden state
    Regenerate
```

### Gate 4: Verification (3B params)

**Problem it solves:**
- Bad code can execute but be wrong
- Model can hallucinate proofs
- Need to catch errors before streaming

**Sub-modules:**

1. **Code Executor (1B):** Sandboxed execution
   - gVisor or firecracker sandbox
   - Test case generation
   - Output comparison
   - Runtime complexity measurement
   - Timeout protection

2. **Proof Checker (1B):** Formal proof validation
   - Lean/Coq/Isabelle integration
   - FOL proof validation
   - Invariant verification
   - Counterexample search

3. **Consistency Checker (1B):** Self-consistency voting
   - Generate same answer 5x with different temperatures
   - Vote on consistency
   - Cross-validate between reasoning steps

**Verdict Combiner:**
```
Input: [code_passed, proof_valid, consistent]
       ↓
MLP → verdict_score (0-1)
       ↓
IF verdict > 0.5: STREAM
ELSE IF fixable: LOOP to Gate 3
ELSE: ABORT, offer alternatives
```

---

## 4. Data Strategy

### Phase 1: Collection (DeepSeek-V4 → Student)

**Sources:**
1. DeepSeek-V4 API responses (primary)
2. Competition math (AMC, AIME, IMO)
3. Code Forces problem solutions
4. LeetCode (filtered for quality)
5. Formal proofs (Lean/Coq)
6. OpenWebMath (reasoning traces)

**Filters (applied post-collection):**
- English only (langdetect, confidence > 0.9)
- Must have code blocks (for coding tasks)
- Must have reasoning (for math tasks)
- Min 100 tokens response length
- Remove duplicate problems

**Dataset size target:** ~2M examples

**Rate limiting:** 60 req/min to avoid API throttling

### Phase 2: SFT Distillation

**Key insight:** We're not just training on (question, answer) pairs.
We're training the *reasoning process*, not just the output.

**Training approach:**
- Standard next-token prediction on full (question, response)
- Labels only on response tokens (not question)
- Low learning rate (1e-5) to preserve pre-training
- Warmup ratio 0.1
- Cosine LR schedule
- Gradient checkpointing for memory

**Efficiency:** LoRA with rank=16, alpha=32
- Only trainable: Q, K, V, O projections
- Freeze everything else during initial SFT

### Phase 3: Gate Training (Sequential)

**Why sequential, not joint?**
- Different gates have different convergence rates
- Joint training leads to interference
- Sequential allows debugging each gate independently

**Order:**
1. Gate 4 (Verification) - most impactful, easiest to validate
2. Gate 1 (Knowledge Retrieval) - next most impactful
3. Gate 2 (Symbolic Reasoning) - complex, needs careful tuning
4. Gate 3 (RL) - last, needs working pipeline first

### Phase 4: RLHF Integration

**GRPO implementation:**
```python
for each batch:
    for each question:
        generate G=8 responses
        compute rewards (execution_correctness, answer_match, style)
        normalize rewards within group
        compute advantages
        PPO update with clipped objectives
```

**Reward shaping:**
- Primary: Code execution passes tests
- Secondary: Answer correctness
- Tertiary: Response quality (readability)

---

## 5. Missing Components (Design Decisions Needed)

### 1. Tool Use Gate (HIGH PRIORITY)
Not in current design but needed:
- Web search for factual queries
- Calculator for math
- File system for code projects
- Shell for command execution

**Recommendation:** Add as Gate 5 after initial 4-gate pipeline works.

### 2. Safety/Constitutional Gate (HIGH PRIORITY)
Content filtering before streaming:
- Toxic content detection
- Harmful code prevention
- PII redaction

**Recommendation:** Lightweight classifier (<100M params), runs before Gate 4.

### 3. Memory Persistence (MEDIUM)
Current design has ephemeral episodic memory.
Need:
- Disk-persisted memory for long-term
- Semantic search over memories
- Memory consolidation during idle time

**Recommendation:** Use FAISS for embedding search, update periodically.

### 4. Quantization Support (HIGH PRIORITY for deployment)
Target: INT8 for production, INT4 for edge
- Current design is FP16/BF16
- Need to validate gates work under quantization
- Per-channel quantization for LM heads
- INT8 inference for gates

**Recommendation:** Post-training quantization after full training.

### 5. Speculative Decoding (MEDIUM)
Faster inference:
- Draft model (7B) generates 4 tokens
- Verify all 4 in parallel with main model
- Accept if all pass verification

**Recommendation:** Use Qwen2.5-Coder-1.5B as draft model.

### 6. Curriculum Learning (MEDIUM)
Progressive difficulty:
- Phase 1: Easy problems (AIMEeasy, LeetCodeeasy)
- Phase 2: Medium problems (AIME, Code Forces Div2)
- Phase 3: Hard problems (IMO, Code Forces Div1)
- Phase 4: Formal proofs

**Recommendation:** Implement as data sampling weights, not separate training.

### 7. Continual Learning (LOW for now)
Ongoing updates without full retrain:
- Elastic weight consolidation
- Knowledge distillation from new data
- Memory replay

**Recommendation:** Not needed until v1 is stable.

---

## 6. Parameter Budget

| Component | Parameters | % of 30B | Notes |
|-----------|-----------|----------|-------|
| Token Embeddings | 1B | 3.3% | Shared, frozen during gate training |
| Base LM (expanded 7B→20B) | 20B | 66.7% | Core language capability |
| Gate 1: Knowledge Retrieval | 2B | 6.7% | Attention + memory + fusion |
| Gate 2: Symbolic Reasoning | 4B | 13.3% | FOL + NatLog + prover |
| Gate 3: RL | 1B | 3.3% | Reward + value + correction |
| Gate 4: Verification | 3B | 10.0% | Executor + checker + consistency |
| **Total** | **31B** | **103.3%** | Some sharing reduces to ~30B |

---

## 7. Training Timeline

| Phase | Duration | GPU Hours | Parallel? |
|-------|----------|-----------|-----------|
| Phase 0: Setup | 1 day | 0 | - |
| Phase 1: Data Collection | 7 days | 0 | API calls |
| Phase 2: SFT | 7 days | 560 | 8x H100 |
| Phase 3: Gate 4 (Verification) | 3 days | 240 | 8x H100 |
| Phase 4: Gate 1 (Knowledge) | 4 days | 320 | 8x H100 |
| Phase 5: Gate 2 (Symbolic) | 5 days | 400 | 8x H100 |
| Phase 6: Gate 3 (RL) | 4 days | 320 | 8x H100 |
| Phase 7: Integration | 3 days | 240 | 8x H100 |
| Phase 8: Evaluation | 2 days | 16 | Single A100 |
| **Total** | **36 days** | **2096 GPU hours** | ~$10.5K on spot |

---

## 8. Key Insights & Design Decisions

### Insight 1: Gates are not independent
Early design thought: "Train each gate independently, compose at inference."
Reality: Gates need joint tuning for composition.

**Solution:** After sequential training, do a joint fine-tuning pass with all gates enabled.

### Insight 2: Verification must be lightweight
Gate 4 (Verification) is in the critical path during token generation.

**Solution:** 
- Use neural approximations for most checks
- Only run full sandboxed execution on complete code blocks
- Async verification (verify previous block while generating next)

### Insight 3: RL needs stable base policy
GRPO/PPO can destabilize if base SFT model isn't solid.

**Solution:**
- Ensure SFT loss is fully converged before RL
- Use conservative KL penalty to stay near base
- Start with high clipping epsilon, reduce over training

### Insight 4: Symbolic reasoning needs formal grounding
The FOL formalizer can't be purely neural.

**Solution:**
- Use LLM to generate FOL formulas
- Validate FOL syntax with parser before reasoning
- Keep symbolic prover separate from neural components

### Insight 5: Memory retrieval is a skill
Initial design had retrieval as a simple attention mechanism.
Reality: Retrieval quality depends heavily on query construction.

**Solution:**
- Learn query transformation before retrieval
- Use multiple retrieval heads for different memory types
- Curriculum: start with high relevance threshold, lower over time

---

## 9. Failure Modes & Mitigations

### Failure: Gate 4 verification slows down generation
**Mitigation:** Async verification, only verify code blocks not text

### Failure: RL destabilizes model
**Mitigation:** Conservative KL penalty, gradual epsilon reduction, value function baseline

### Failure: Memory retrieval returns irrelevant info
**Mitigation:** Relevance scoring gate, importance-weighted replacement, query transformation

### Failure: Symbolic reasoning produces invalid FOL
**Mitigation:** Syntax validation layer, fall back to neural reasoning if FOL parsing fails

### Failure: Model generates unsafe content
**Mitigation:** Constitutional gate before verification, content classifiers on output

### Failure: Knowledge distillation collapses to copying teacher
**Mitigation:** Regularization to prevent mode collapse, diverse sampling, KL divergence penalty

---

## 10. Evaluation Strategy

### Benchmarks
| Benchmark | Target | Current (Qwen2.5-Coder-7B) |
|-----------|--------|----------------------------|
| HumanEval | 85% | 88.4% (already exceeds!) |
| MBPP | 80% | 82.1% (already exceeds!) |
| MATH | 55% | 51.2% (need improvement) |
| ARC-Challenge | 60% | ~40% (need work) |
| MMLU | 75% | 70.2% (need work) |

### Evaluation Protocol
1. **Offline:** Run full benchmark suite weekly
2. **Online:** Track verification pass rate, self-correction rate
3. **A/B:** Compare gated vs non-gated on held-out problems
4. **Human:** Manual review of proofs and complex solutions

### Key Metrics to Track
- Verification pass rate (Gate 4 effectiveness)
- Self-correction rate (Gate 3 effectiveness)
- Memory retrieval hit rate (Gate 1 effectiveness)
- Proof validity rate (Gate 2 effectiveness)
- Token-level confidence calibration
- Inference latency (ms/token)

---

## 11. Open Questions

1. **Should we expand to 30B or stay at 20B?**
   - 30B gives more gate capacity
   - 20B is faster and cheaper
   - Need to test quality vs cost tradeoff

2. **Gate training order: is Verification → Knowledge → Symbolic → RL correct?**
   - Could start with Knowledge (easiest to validate)
   - Or Symbolic (most novel, needs most tuning)
   - Need ablation study

3. **How to handle non-English queries?**
   - Current scope is English-only
   - But users may send non-English
   - Option: Fall back to base model for non-English
   - Option: Add translation gate

4. **When to switch from SFT to RL?**
   - Current: SFT → Gate training → RL
   - Alternative: Interleave SFT and RL
   - Need to test sample efficiency

5. **Formal proofs: which prover?**
   - Lean 4 (modern, good tooling)
   - Coq (mature, many proofs)
   - Isabelle (good for higher-order logic)
   - Recommendation: Start with Lean 4

6. **Memory: how often to update?**
   - Online (every query)
   - Batch (end of training epoch)
   - Periodic (daily consolidation)
   - Recommendation: Batch with importance filtering

---

## 12. Future Directions

### v1.1 (After baseline works)
- Add Tool Use Gate (web search, calculator, shell)
- Implement speculative decoding for 2x speedup
- Add safety constitutional gate

### v1.2 (Quality improvements)
- Curriculum learning pipeline
- Continual learning for memory updates
- Multi-task training (code + math + reasoning jointly)

### v2.0 (Major upgrade)
- Expand to 70B base model
- Add vision encoding for diagram understanding
- Multi-modal reasoning
- Agentic planning across long horizons

---

## 13. Repository Structure

```
distill-v4/
├── README.md                    # Project overview
├── DESIGN.md                   # This document
├── docs/
│   ├── ARCHITECTURE.md         # Technical architecture
│   ├── SEED_MODEL_SELECTION.md # Seed model analysis
│   └── DISTILLATION_PIPELINE.md # Training pipeline
├── configs/                    # Training configs
├── scripts/                   # Data collection, training, eval
├── src/
│   ├── gates/                 # Gate module implementations
│   ├── inference/             # Full inference pipeline
│   ├── training/              # Training loops
│   ├── models/                # Base model modifications
│   ├── data/                  # Data processing
│   ├── eval/                  # Evaluation harness
│   └── utils/                 # Utilities
├── data/                      # Data directories
│   ├── raw/                   # Raw API responses
│   ├── processed/             # Filtered, tokenized
│   ├── sft/                   # SFT training data
│   ├── rl/                    # RL training data
│   └── eval/                  # Evaluation benchmarks
├── checkpoints/              # Model checkpoints
├── logs/                      # Training logs
└── requirements.txt
```

---

*Document generated: 2026-06-06*
*Last updated: 2026-06-06*
*Author: Swaroop Kallakuri (with assistance from Hermes Agent)*