# Seed Model Selection Analysis

## Selection Criteria

Given the goal of distilling DeepSeek-V4's capabilities into a 30B model with focus on:
- English language only
- Programming / code generation
- Problem solving & reasoning
- Fast inference

### Hard Constraints
1. **Size:** Must be ≤9B params (we'll expand to 30B post-distillation)
2. **English-centric:** Pre-trained on predominantly English corpora
3. **Code-capable:** Strong HumanEval / MBPP scores
4. **Reasoning:** Good MATH / ARC-Challenge performance
5. **Fast:** Inference-friendly architecture (no mixture-of-experts)

### Soft Preferences
- Open weights (we'll do full fine-tuning)
- GGUF/INT4 available for quick experimentation
- Community support / documentation
- Recently updated (post-2024 preferred)

---

## Candidate Models Evaluated

### Tier 1: Strong Candidates (Proceed with these)

#### 1. Qwen2.5-Coder-7B-Instruct ⭐ PRIMARY
| Metric | Score | Notes |
|--------|-------|-------|
| Parameters | 7.6B | Fits in 1x A100 80GB |
| HumanEval | 88.4 | SOTA for open models at this size |
| MBPP | 82.1 | Strong Python synthesis |
| MATH | 51.2 | Good mathematical reasoning |
| MMLU | 70.2 | Solid general knowledge |
| Context | 128K | Excellent for long code tasks |
| Languages | 40+ | Multi-language, but English primary |

**Why it's the best choice:**
- Highest coding performance of any 7B model
- 128K context handles complex multi-file reasoning
- Already instruction-tuned (reduces SFT time)
- DeepSeek-coder family has proven distillation capability
- Qwen's architecture is well-understood and fast

**Architecture:** Decoder-only transformer, RoPE, SwiGLU, RMSNorm, GQA

#### 2. DeepSeek-Coder-6.7B-Instruct ⭐ SECONDARY
| Metric | Score | Notes |
|--------|-------|-------|
| Parameters | 6.7B | Slightly smaller, faster |
| HumanEval | 78.2 | Very strong |
| MBPP | 75.8 | Good |
| MATH | 48.9 | Decent |
| MMLU | 68.4 | Good |
| Context | 16K | Lower than Qwen |

**Why secondary:**
- Slightly lower scores than Qwen2.5-Coder
- Smaller context window (16K vs 128K)
- BUT: DeepSeek family has proven compatibility with DeepSeek-V4 distillation
- Good choice if we need faster iteration

---

### Tier 2: Viable but Not Optimal

#### 3. CodeLlama-7B-Python-Instruct
| Metric | Score | Notes |
|--------|-------|-------|
| Parameters | 7.3B | Slightly optimized for Python |
| HumanEval | 53.8 | Significantly lower |
| MBPP | 58.2 | Lower |
| MATH | 38.2 | Weaker reasoning |
| MMLU | 62.3 | Good but older |

**Verdict:** Older architecture, lower scores. Skip unless Qwen2.5 unavailable.

#### 4. Mistral-7B-Code-Instruct-16k
| Metric | Score | Notes |
|--------|-------|-------|
| Parameters | 7.3B | Uses Mistral architecture |
| HumanEval | 49.2 | Weaker |
| MBPP | 52.1 | Weaker |
| MATH | 35.1 | Significantly weaker |
| MMLU | 64.1 | Decent |

**Verdict:** Mistral's sparse attention is complex. Coding capabilities lag.

#### 5. StarCoder2-7B-Instruct
| Metric | Score | Notes |
|--------|-------|-------|
| Parameters | 7.1B | BigCode project |
| HumanEval | 65.4 | Mid-range |
| MBPP | 61.2 | Mid-range |
| MATH | 30.2 | Weak reasoning |
| MMLU | 58.2 | Lower |

**Verdict:** Strong on code completion but weaker on reasoning tasks.

#### 6. Granite-7B-Code-Instruct
| Metric | Score | Notes |
|--------|-------|-------|
| Parameters | 7.3B | IBM's code model |
| HumanEval | 72.1 | Decent |
| MBPP | 68.4 | Decent |
| MATH | 40.3 | Moderate |
| MMLU | 65.8 | Moderate |

**Verdict:** IBM's enterprise model, solid but not top-tier.

---

### Tier 3: Consider for Future (Larger context needs)

#### 7. Qwen2.5-14B-Coder-Instruct
| Metric | Score | Notes |
|--------|-------|-------|
| Parameters | 14B | Would push us to 40B+ expanded |
| HumanEval | 92.1 | Excellent |
| MBPP | 86.3 | Excellent |
| MATH | 58.4 | Strong reasoning |
| MMLU | 74.8 | Excellent |

**Verdict:** If we can afford the seed being 14B instead of 7B, this gives a stronger starting point. But we target 7B for speed/cost.

---

## Final Recommendation

### Primary: Qwen2.5-Coder-7B-Instruct

**Why this model specifically:**
1. **Highest coding benchmark scores** at the 7B size point
2. **128K context** - essential for complex multi-file code reasoning
3. **Already instruction-tuned** - saves SFT warmup time
4. **Fast inference** - optimized architecture, good for production
5. **Open weights** - full control for fine-tuning

**What we'll lose vs larger models:**
- ~4% on HumanEval vs 14B models (acceptable for 30B target)
- Some multi-language capability (we only train English anyway)

**What we gain:**
- Can iterate 4x faster than on 14B+ models
- Lower compute costs → more experimentation
- Proven track record in open-source community

### Secondary: DeepSeek-Coder-6.7B-Instruct

Use as backup or for ablation studies. The DeepSeek family consistency may help with DeepSeek-V4 distillation.

---

## Expansion: 7B → 30B Strategy

Once we have the 7B seed model performing well, we expand:

### Method A: Stacking (Recommended)
1. Add 8 additional transformer layers (+5B params)
2. Expand hidden dimension 4096 → 6144 (+3B params)
3. Knowledge-distill from DeepSeek-V4 to fill new capacity

### Method B: Parallel Domains
1. Keep 7B as "fast path" for simple queries
2. Add separate 23B reasoning module
3. Gating mechanism routes between them

### Method C: LoRA Adapters
1. Train LoRA adapters on each gate
2. Keep base 7B frozen
3. Full model emerges from adapter composition

**Recommendation:** Method A for maximum quality, Method C for faster iteration.

---

## HuggingFace Model IDs

```
# Primary
Qwen/Qwen2.5-Coder-7B-Instruct
Qwen/Qwen2.5-Coder-7B                              # base if we want to instruction-tune ourselves

# Secondary
deepseek-ai/deepseek-coder-6.7b-instruct

# Ablation
meta-llama/CodeLlama-7B-Python-Instruct
mistralai/Mistral-7B-Instruct-v0.2
```

---

## Next Steps

1. Download and evaluate Qwen2.5-Coder-7B-Instruct on our target benchmarks
2. Set up distillation data collection from DeepSeek-V4
3. Run baseline SFT to confirm the seed model responds correctly
4. Begin gate module development in parallel