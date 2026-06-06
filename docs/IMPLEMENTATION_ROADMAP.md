# Distill-V4 Implementation Roadmap

## Targets

| Benchmark | Current (Qwen2.5-Coder-7B) | Phase 3 Goal | Final Goal |
|-----------|---------------------------|--------------|------------|
| **HumanEval** | 88.4% | 90% | 92% |
| **MATH** | 51.2% | 56% | 60% |
| **MBPP** | 82.1% | 86% | 90% |
| **MMLU** | 70.2% | 74% | 78% |
| **GSM8K** | 83.4% | 88% | 92% |
| **BBH** | 71.3% | 76% | 80% |

**Final model: 30B parameters, 4-gate architecture, English-only, pre-verification before streaming.**

---

## Phase 0: Infrastructure Setup (Week 1-2)

### 0.1 Environment & Compute
- [ ] **GPU Cluster Setup**
  - [ ] Minimum: 8x A100 80GB (for 30B full training)
  - [ ] Alternative: 16x A100 40GB (gradient accumulation)
  - [ ] Setup SLURM or Kubernetes job scheduling
  - [ ] Verify CUDA 12.1+, PyTorch 2.3+, TransformerLens

- [ ] **Data Storage**
  - [ ] S3/GCS bucket for raw distillation data (~500GB)
  - [ ] Local NVMe for processed training data (~200GB)
  - [ ] Model checkpoints: 60GB free (30B FP16 = 60GB)

- [ ] **Software Stack**
  - [ ] Python 3.11+, venv/conda
  - [ ] PyTorch 2.3+ with CUDA 12.1
  - [ ] HuggingFace Transformers + PEFT
  - [ ] DeepSpeed ZeRO-2/3 (for 30B training)
  - [ ] Weights & Biases (experiment tracking)
  - [ ] Docker image with all dependencies

### 0.2 Repository Setup
```bash
git clone https://github.com/ksjpswaroop/distill-v4.git
cd distill-v4
# Verify structure
ls -la
# docs/ src/ configs/ scripts/ data/ models/
```

### 0.3 Seed Model Acquisition
```bash
# PRIMARY: Qwen2.5-Coder-7B-Instruct
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct --local-dir ./models/Qwen2.5-Coder-7B-Instruct

# BACKUP: DeepSeek-Coder-6.7B
huggingface-cli download deepseek-ai/DeepSeek-Coder-6.7B-Instruct --local-dir ./models/DeepSeek-Coder-6.7B-Instruct

# Verify model loads
python -c "from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('./models/Qwen2.5-Coder-7B-Instruct'); print(f'Model loaded: {m.num_parameters()} params')"
```

### 0.4 Baseline Evaluation (Before Any Training)
```bash
# Run full eval suite on seed model
python scripts/evaluate.py \
  --model ./models/Qwen2.5-Coder-7B-Instruct \
  --tasks humaneval,mbpp,math,mmlu,gsm8k,bbh \
  --output results/baseline_seed.json

# Expected results:
# HumanEval: 88.4%, MBPP: 82.1%, MATH: 51.2%
# MMLU: 70.2%, GSM8K: 83.4%, BBH: 71.3%
```

**Gate 0: Record baseline. Do NOT proceed until baseline is confirmed.**

---

## Phase 1: Data Collection & Processing (Week 2-4)

### 1.1 Distillation Data Sources
| Source | Type | Volume | Priority |
|--------|------|--------|----------|
| DeepSeek-V4 API | Synthetic (question → DeepSeek answer) | 500K | P0 |
| OpenAI o3-mini API | Synthetic (reasoning traces) | 200K | P1 |
| Anthropic Claude API | Synthetic (reasoning traces) | 200K | P1 |
| OpenMathInstruct | Math reasoning | 400K | P0 |
| PrimeIntellect/Reed-SFT | Synthetic math/code | 300K | P1 |
| Magpie-CoT | Chain-of-thought pairs | 500K | P0 |
| Custom Python scripts | Code generation | 200K | P1 |

### 1.2 Data Collection Pipeline
```python
# scripts/collect_distillation_data.py (already exists — verify it works)
python scripts/collect_distillation_data.py \
  --source deepseek \
  --output ./data/raw/deepseek_responses.jsonl \
  --num_samples 100000 \
  --categories coding,math,logic,reasoning

# Validate data format
python -c "
import json
with open('./data/raw/deepseek_responses.jsonl') as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        assert 'question' in d and 'answer' in d, f'Line {i} invalid'
        assert len(d['answer']) > 50, f'Line {i} too short'
        if i >= 9: break
print('Data format validated: 10 samples OK')
"
```

### 1.3 English Language Filter
```bash
# Filter non-English content (exclude Chinese, Spanish, French, etc.)
python scripts/filter_english.py \
  --input ./data/raw/all_responses.jsonl \
  --output ./data/processed/english_only.jsonl \
  --min-english-ratio 0.95

# Verify filter
python -c "
import json
with open('./data/processed/english_only.jsonl') as f:
    en_count = sum(1 for _ in f)
with open('./data/raw/all_responses.jsonl') as f:
    total_count = sum(1 for _ in f)
print(f'English: {en_count}/{total_count} = {en_count/total_count*100:.1f}%')
"
```

### 1.4 Deduplication
```bash
# Deduplicate by question hash (avoid train-test contamination)
python scripts/deduplicate.py \
  --input ./data/processed/english_only.jsonl \
  --output ./data/processed/deduped.jsonl \
  --hash_field question

# Verify no duplicates
python -c "
import json, hashlib
seen = set()
dupes = 0
with open('./data/processed/deduped.jsonl') as f:
    for line in f:
        h = hashlib.sha256(json.loads(line)['question'].encode()).hexdigest()
        if h in seen: dupes += 1
        seen.add(h)
print(f'Duplicates after dedup: {dupes}')
"
```

### 1.5 Data Split
```bash
# 90% train, 5% eval, 5% test (no overlap)
python scripts/train_test_split.py \
  --input ./data/processed/deduped.jsonl \
  --output_dir ./data/splits \
  --train 0.90 --eval 0.05 --test 0.05 \
  --stratify category

ls -lh ./data/splits/
# train.jsonl (~900K), eval.jsonl (~50K), test.jsonl (~50K)
```

### 1.6 Quality Filtering
```bash
# Remove samples where answer is too short, has toxic content, or low quality
python scripts/quality_filter.py \
  --input ./data/splits/train.jsonl \
  --output ./data/splits/train_filtered.jsonl \
  --min_answer_len 100 \
  --max_answer_len 8192 \
  --remove_toxic true \
  --remove_incomplete true

# Final count verification
wc -l ./data/splits/train_filtered.jsonl
# Expected: ~850K-900K high-quality samples
```

**Gate 1: Minimum 500K filtered samples. Data quality verified. Proceed to Phase 2.**

---

## Phase 2: Supervised Fine-Tuning — Base Model (Week 4-8)

### 2.1 SFT Training Configuration
```yaml
# configs/sft_config.yaml
model:
  name: ./models/Qwen2.5-Coder-7B-Instruct
  seq_length: 8192
  gradient_checkpointing: true

training:
  batch_size: 8  # per GPU, total 64 with 8 GPUs
  gradient_accumulation_steps: 8
  learning_rate: 1e-5
  warmup_ratio: 0.1
  num_train_epochs: 3
  weight_decay: 0.01
  max_grad_norm: 1.0
  fp16: false
  bf16: true
  
optimizer:
  type: AdamW
  beta1: 0.9
  beta2: 0.999

scheduler:
  type: cosine
  min_lr_ratio: 0.1

deepspeed:
  stage: 2  # ZeRO-2 for 30B
  offload_optimizer: false
```

### 2.2 Launch SFT Training
```bash
# Single command launch
deepspeed --num_gpus=8 \
  src/training/sft_train.py \
  --config configs/sft_config.yaml \
  --data_path ./data/splits/train_filtered.jsonl \
  --eval_data ./data/splits/eval.jsonl \
  --output_dir ./checkpoints/sft_base_7b \
  --wandb_project distill-v4-sft \
  --wandb_run_name sft-base-qwen-7b

# Expected training time: ~72 hours on 8x A100 80GB
# Total steps: ~30,000 (850K samples / 64 batch / 3 epochs)
```

### 2.3 Monitor Training
```bash
# Watch W&B dashboard
# Key metrics to monitor:
# - train/loss: should decrease from ~2.5 to ~0.8
# - eval/loss: should not diverge (overfitting check)
# - eval/humaneval: should increase from 88.4% baseline

# If eval/loss increases while train/loss decreases → overfitting, stop early
```

### 2.4 Checkpoint Selection
```bash
# List checkpoints
ls -la ./checkpoints/sft_base_7b/

# Load each checkpoint and eval
for ckpt in ./checkpoints/sft_base_7b/checkpoint-*/; do
  echo "Evaluating $ckpt"
  python scripts/evaluate.py \
    --model $ckpt \
    --tasks humaneval,mbpp,math \
    --output "results/ckpt_eval_$(basename $ckpt).json"
done

# Select best checkpoint (highest harmonic mean of HumanEval + MBPP + MATH)
python scripts/select_best_checkpoint.py \
  --results_dir results/ \
  --metric harmonic_mean \
  --output best_sft_checkpoint.txt
```

### 2.5 Post-SFT Evaluation
```bash
# Full evaluation on held-out test set
python scripts/evaluate.py \
  --model ./checkpoints/sft_base_7b/best/ \
  --tasks humaneval,mbpp,math,mmlu,gsm8k,bbh \
  --output results/test_sft_base.json

# Expected improvements over baseline:
# HumanEval: 88.4% → 90-91%
# MBPP: 82.1% → 85-86%
# MATH: 51.2% → 55-57%
```

**Gate 2: SFT model achieves 90%+ HumanEval, 55%+ MATH on test set. Proceed to Phase 3.**

---

## Phase 3: Gate Module Training (Week 8-14)

### 3.1 Gate 1 — Knowledge Retrieval Gate (2B params)

#### Architecture
```
Input: hidden_states [batch, seq, 4096]
       + query embedding
       + episodic memory bank (Faiss index)

Output: retrieved_context + gated_hidden_states
```

#### Training Data
```bash
# Create retrieval training pairs
python scripts/prepare_retrieval_data.py \
  --input ./data/splits/train_filtered.jsonl \
  --output ./data/gates/retrieval_train.jsonl \
  --task-type fact_lookup,code_reference,documentation

# Create Faiss index from training corpus
python scripts/build_faiss_index.py \
  --documents ./data/splits/train_filtered.jsonl \
  --output ./indices/knowledge_index.faiss
```

#### Training
```bash
deepspeed --num_gpus=4 \
  src/gates/train_retrieval_gate.py \
  --base_model ./checkpoints/sft_base_7b/best/ \
  --gate_data ./data/gates/retrieval_train.jsonl \
  --index_path ./indices/knowledge_index.faiss \
  --output ./checkpoints/gate1_retrieval \
  --gate_params 2B \
  --learning_rate 5e-5 \
  --num_epochs 2
```

#### Evaluation
```bash
python scripts/evaluate_gate.py \
  --model ./checkpoints/gate1_retrieval/best/ \
  --gate retrieval \
  --tasks factuality,natural_questions,triviaqa \
  --output results/gate1_retrieval_eval.json

# Expected: +5-8% on factuality tasks
```

### 3.2 Gate 2 — Symbolic Reasoning (FOL) Gate (4B params)

#### Architecture
```
Input: hidden_states [batch, seq, 4096]
       + FOL formalization of problem

Processing:
  - FOL encoder (transformer-based)
  - Natural logic reasoner
  - Theorem prover (modified HTPS)

Output: reasoned_hidden_states + proof_trace
```

#### Training Data
```bash
# Create FOL reasoning pairs
python scripts/prepare_fol_data.py \
  --input ./data/splits/train_filtered.jsonl \
  --output ./data/gates/fol_train.jsonl \
  --categories logic,math,proofs,reasoning

# Categories:
# - FOL formalization (English → FOL)
# - Proof generation (FOL → natural language proof)
# - Theorem proving (hypothesis → proof/non-proof)
```

#### Training
```bash
deepspeed --num_gpus=8 \
  src/gates/train_fol_gate.py \
  --base_model ./checkpoints/sft_base_7b/best/ \
  --gate_data ./data/gates/fol_train.jsonl \
  --output ./checkpoints/gate2_fol \
  --gate_params 4B \
  --learning_rate 3e-5 \
  --num_epochs 3
```

#### Evaluation
```bash
python scripts/evaluate_gate.py \
  --model ./checkpoints/gate2_fol/best/ \
  --gate fol \
  --tasks fol_benchmarks,proof_success,logical_equilibrium \
  --output results/gate2_fol_eval.json

# Expected: +10-15% onFOL benchmarks, 78%+ proof success on simple theorems
```

### 3.3 Gate 3 — Reinforcement Learning Gate (1B params)

#### Architecture
```
Input: hidden_states [batch, seq, 4096]
       + reward signals

Processing:
  - Reward model (1B)
  - PPO/GRPO policy update
  - Value baseline

Output: rl_boosted_hidden_states + confidence_score
```

#### Training (GRPO — DeepSeek-R1 method)
```bash
# Collect reward signals
python scripts/compute_rewards.py \
  --model ./checkpoints/sft_base_7b/best/ \
  --data ./data/splits/train_filtered.jsonl \
  --reward_type accuracy,consistency,halting \
  --output ./data/rl/reward_signals.jsonl

# GRPO training
deepspeed --num_gpus=4 \
  src/gates/train_rl_gate.py \
  --base_model ./checkpoints/sft_base_7b/best/ \
  --reward_data ./data/rl/reward_signals.jsonl \
  --output ./checkpoints/gate3_rl \
  --gate_params 1B \
  --algorithm grpo \
  --kl_coef 0.04 \
  --num_iterations 100
```

#### Evaluation
```bash
python scripts/evaluate_gate.py \
  --model ./checkpoints/gate3_rl/best/ \
  --gate rl \
  --tasks alignment,helpfulness,safety \
  --output results/gate3_rl_eval.json

# Expected: +5-8% on alignment benchmarks, stable safety scores
```

### 3.4 Gate 4 — Verification Gate (3B params)

#### Architecture
```
Input: hidden_states [batch, seq, 4096]
       + candidate response

Processing (pre-streaming):
  - Code executor (sandbox)
  - Formal proof checker
  - Consistency validator (cross-reference)
  - Hallucination detector

Output: verified_hidden_states + confidence_score + rejection_flag
```

#### Training Data
```bash
# Create verification pairs (correct + incorrect examples)
python scripts/prepare_verification_data.py \
  --input ./data/splits/train_filtered.jsonl \
  --output ./data/gates/verification_train.jsonl \
  --include_wrong_answers true \
  --wrong_answer_ratio 0.3

# Categories:
# - Correct code execution
# - Incorrect code with bugs (to train rejection)
# - Formal proofs (valid + invalid)
# - Math answers (correct + incorrect with explanation why)
```

#### Training
```bash
deepspeed --num_gpus=4 \
  src/gates/train_verification_gate.py \
  --base_model ./checkpoints/sft_base_7b/best/ \
  --gate_data ./data/gates/verification_train.jsonl \
  --output ./checkpoints/gate4_verification \
  --gate_params 3B \
  --learning_rate 5e-5 \
  --num_epochs 2
```

#### Evaluation
```bash
python scripts/evaluate_gate.py \
  --model ./checkpoints/gate4_verification/best/ \
  --gate verification \
  --tasks code_execution,proof_checking,consistency \
  --output results/gate4_verification_eval.json

# Expected: 95%+ correct acceptance, 80%+ correct rejection of wrong answers
```

**Gate 3: All 4 gates trained and individually evaluated. Proceed to Phase 4.**

---

## Phase 4: Model Merging & Full Integration (Week 14-16)

### 4.1 Gate Merging Strategy
```python
# src/training/merge_gates.py

# Option A: Sequential (our primary)
# Base → Retrieval → FOL → RL → Verification → Stream

# Option B: Parallel (ensemble)
# Base → [Retrieval, FOL, RL] → Verification → Stream
# Attention-based routing between parallel gates

# Option C: LoRA adaptation (recommended for stability)
# Base + LoRA(Retrieval) + LoRA(FOL) + LoRA(RL) + LoRA(Verification)
```

### 4.2 Merge Gates
```bash
python src/training/merge_gates.py \
  --base ./checkpoints/sft_base_7b/best/ \
  --gate1 ./checkpoints/gate1_retrieval/best/ \
  --gate2 ./checkpoints/gate2_fol/best/ \
  --gate3 ./checkpoints/gate3_rl/best/ \
  --gate4 ./checkpoints/gate4_verification/best/ \
  --merge_strategy lora \
  --output ./checkpoints/full_model_30b

# This creates the 30B model:
# Base: 20B + Gate1: 2B + Gate2: 4B + Gate3: 1B + Gate4: 3B = 30B
```

### 4.3 Joint Fine-Tuning (Light)
```bash
# Fine-tune merged model on a small subset to improve gate coordination
deepspeed --num_gpus=8 \
  src/training/joint_finetune.py \
  --model ./checkpoints/full_model_30b \
  --data ./data/splits/train_filtered.jsonl \
  --output ./checkpoints/joint_finetuned_30b \
  --num_epochs 1 \
  --learning_rate 5e-6 \
  --batch_size 4  # smaller batch for merged model
```

### 4.4 Full Evaluation
```bash
# Complete evaluation suite
python scripts/evaluate.py \
  --model ./checkpoints/joint_finetuned_30b/best/ \
  --tasks humaneval,mbpp,math,mmlu,gsm8k,bbh,bigcodebench,livecodebench \
  --output results/final_model_eval.json \
  --temperature 0.2 \
  --num_samples 1000

# TARGETS CHECK:
# HumanEval: 92%+ (currently 88.4% baseline)
# MATH: 60%+ (currently 51.2% baseline)
# MBPP: 90%+ (currently 82.1% baseline)
# MMLU: 78%+ (currently 70.2% baseline)
# GSM8K: 92%+ (currently 83.4% baseline)
# BBH: 80%+ (currently 71.3% baseline)
```

### 4.5 Gate Interaction Analysis
```bash
# Verify gates are actually being used
python scripts/analyze_gate_usage.py \
  --model ./checkpoints/joint_finetuned_30b/best/ \
  --test_data ./data/splits/test.jsonl \
  --output results/gate_usage_analysis.json

# Metrics:
# - retrieval_gate_activation_rate: should be > 30%
# - fol_gate_activation_rate: should be > 20%
# - rl_gate_activation_rate: should be > 40%
# - verification_rejection_rate: should be 5-15%
```

**Gate 4: Final model meets all targets. Proceed to Phase 5.**

---

## Phase 5: Inference Optimization (Week 16-18)

### 5.1 Quantization
```bash
# Quantize to INT8 for deployment
python src/inference/quantize.py \
  --model ./checkpoints/joint_finetuned_30b/best/ \
  --output ./models/distill-v4-30b-int8 \
  --quant_type int8 \
  --technique gptq

# Quantize to INT4 for edge deployment (optional)
python src/inference/quantize.py \
  --model ./checkpoints/joint_finetuned_30b/best/ \
  --output ./models/distill-v4-30b-int4 \
  --quant_type int4 \
  --technique ggml
```

### 5.2 Inference Benchmarking
```bash
# Benchmark latency/throughput
python src/inference/benchmark.py \
  --model ./models/distill-v4-30b-int8 \
  --tasks humaneval \
  --batch_sizes 1,4,16 \
  --num_samples 100 \
  --output results/benchmark_int8.json

# Target:
# Throughput: > 50 tokens/sec on 1x A100
# Latency p50: < 200ms per token
# Memory: < 40GB for INT8
```

### 5.3 Streaming Verification
```bash
# Test pre-verification streaming
python src/inference/stream_test.py \
  --model ./models/distill-v4-30b-int8 \
  --prompt "Write a Python function to reverse a linked list" \
  --verify_before_stream true \
  --output results/stream_test.json
```

### 5.4 API Server
```bash
# Launch inference server
python src/inference/api_server.py \
  --model ./models/distill-v4-30b-int8 \
  --port 8080 \
  --max_batch_size 16

# Test endpoint
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "def quicksort(arr):", "max_tokens": 200}'
```

---

## Phase 6: Deployment & Monitoring (Week 18-20)

### 6.1 Model Registry
```bash
# Push to HuggingFace Hub
huggingface-cli login
python scripts/push_to_hub.py \
  --model_dir ./models/distill-v4-30b-int8 \
  --repo_id ksjpswaroop/distill-v4-30b \
  --private false \
  --tags "coding,reasoning,distillation,30b"

# Create model card
python scripts/generate_model_card.py \
  --model_path ./models/distill-v4-30b-int8 \
  --benchmark_results results/final_model_eval.json \
  --output ./models/distill-v4-30b-int8/README.md
```

### 6.2 CI/CD for Evaluation
```yaml
# .github/workflows/eval.yml
name: Weekly Eval
on:
  schedule:
    - cron: '0 9 * * 0'  # Every Sunday 9am
  workflow_dispatch:

jobs:
  eval:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - name: Run evaluation
        run: |
          python scripts/evaluate.py \
            --model ./models/distill-v4-30b-int8 \
            --tasks humaneval,mbpp,math \
            --output results/weekly_eval.json
      - name: Check regressions
        run: |
          python scripts/check_regression.py \
            --current results/weekly_eval.json \
            --baseline results/final_model_eval.json \
            --threshold 2.0  # 2% regression tolerance
```

### 6.3 Monitoring Dashboard
```bash
# Deploy W&B dashboard
python scripts/deploy_monitoring.py \
  --wandb_project distill-v4-production \
  --dashboards accuracy,latency,gate_usage
```

---

## Phase 7: Continuous Improvement (Ongoing)

### 7.1 Active Learning
```bash
# Identify weak areas
python scripts/identify_weakness.py \
  --eval_results results/final_model_eval.json \
  --output data/weakness_areas.json

# Generate targeted data for weak areas
python scripts/generate_targeted_data.py \
  --weakness_areas data/weakness_areas.json \
  --output data/retrain/weakness_train.jsonl \
  --num_samples 50000
```

### 7.2 Red Team Testing
```bash
# Adversarial testing
python scripts/red_team.py \
  --model ./models/distill-v4-30b-int8 \
  --categories jailbreak,prompt_injection,toxic \
  --num_tests 1000 \
  --output results/red_team.json

# If issues found, apply safety fine-tuning
```

### 7.3 Model Updates
```bash
# Quarterly re-distillation from updated DeepSeek
# Merge new teacher knowledge while preserving gate improvements
python src/training/incremental_distill.py \
  --student ./models/distill-v4-30b-int8 \
  --teacher deepseek-v4-latest \
  --output ./models/distill-v4-30b-v2
```

---

## Verification Gates Summary

| Gate | Criteria | Exit Threshold |
|------|----------|----------------|
| **Gate 0** | Baseline evaluation | 88.4% HumanEval confirmed |
| **Gate 1** | Data collection complete | 500K+ samples, English-only verified |
| **Gate 2** | SFT training complete | 90%+ HumanEval, 55%+ MATH |
| **Gate 3** | All 4 gates trained | Each gate meets individual targets |
| **Gate 4** | Final model integrated | All 6 benchmarks meet targets |
| **Gate 5** | Inference optimized | Latency < 200ms, throughput > 50 tok/s |
| **Gate 6** | Deployed & monitored | API healthy, no regressions |

---

## File Checklist

```
distill-v4/
├── configs/
│   ├── config.example.yaml
│   ├── sft_config.yaml          # Phase 2
│   ├── gate1_retrieval.yaml     # Phase 3.1
│   ├── gate2_fol.yaml           # Phase 3.2
│   ├── gate3_rl.yaml            # Phase 3.3
│   ├── gate4_verification.yaml  # Phase 3.4
│   └── joint_finetune.yaml      # Phase 4.3
├── data/
│   ├── raw/                     # Phase 1
│   ├── processed/               # Phase 1
│   ├── splits/                  # Phase 1
│   ├── gates/                   # Phase 3
│   └── rl/                      # Phase 3.3
├── scripts/
│   ├── collect_distillation_data.py  # Phase 1
│   ├── filter_english.py             # Phase 1.3
│   ├── deduplicate.py                # Phase 1.4
│   ├── quality_filter.py             # Phase 1.6
│   ├── evaluate.py                    # Gate 0, 2, 4
│   ├── evaluate_gate.py               # Phase 3
│   ├── select_best_checkpoint.py      # Phase 2.4
│   ├── prepare_retrieval_data.py      # Phase 3.1
│   ├── prepare_fol_data.py            # Phase 3.2
│   ├── prepare_verification_data.py   # Phase 3.4
│   ├── compute_rewards.py             # Phase 3.3
│   ├── merge_gates.py                 # Phase 4.1
│   ├── analyze_gate_usage.py          # Phase 4.5
│   ├── quantize.py                    # Phase 5.1
│   ├── benchmark.py                    # Phase 5.2
│   ├── push_to_hub.py                 # Phase 6.1
│   └── generate_model_card.py         # Phase 6.1
├── src/
│   ├── gates/
│   │   ├── gates.py                # Gate architecture
│   │   ├── retrieval_gate.py         # Phase 3.1
│   │   ├── fol_gate.py              # Phase 3.2
│   │   ├── rl_gate.py               # Phase 3.3
│   │   └── verification_gate.py      # Phase 3.4
│   ├── training/
│   │   ├── sft_train.py             # Phase 2
│   │   ├── train_retrieval_gate.py  # Phase 3.1
│   │   ├── train_fol_gate.py        # Phase 3.2
│   │   ├── train_rl_gate.py         # Phase 3.3
│   │   ├── train_verification_gate.py # Phase 3.4
│   │   ├── merge_gates.py           # Phase 4.1
│   │   └── joint_finetune.py        # Phase 4.3
│   ├── inference/
│   │   ├── pipeline.py              # Inference pipeline
│   │   ├── quantize.py              # Phase 5.1
│   │   ├── benchmark.py             # Phase 5.2
│   │   ├── stream_test.py           # Phase 5.3
│   │   └── api_server.py            # Phase 5.4
│   └── utils/
│       ├── data_utils.py
│       ├── eval_utils.py
│       └── memory/
├── models/
│   ├── Qwen2.5-Coder-7B-Instruct/  # Seed model
│   ├── distill-v4-30b-int8/         # Final model
│   └── checkpoints/                 # All checkpoints
├── docs/
│   ├── ARCHITECTURE.md
│   ├── SEED_MODEL_SELECTION.md
│   ├── DISTILLATION_PIPELINE.md
│   ├── PRIOR-ART.md
│   └── IMPLEMENTATION_ROADMAP.md    # This file
└── README.md
```

---

## Timeline

```
Week  1-2: Phase 0 — Infrastructure
Week  2-4: Phase 1 — Data Collection
Week  4-8: Phase 2 — SFT Training
Week  8-14: Phase 3 — Gate Training (4 gates)
Week 14-16: Phase 4 — Merging & Integration
Week 16-18: Phase 5 — Inference Optimization
Week 18-20: Phase 6 — Deployment
Ongoing:    Phase 7 — Continuous Improvement
```

**Total: ~20 weeks (5 months) from start to production deployment.**