#!/bin/bash
# =============================================================================
# Distill-V4 Full Training Pipeline
# =============================================================================
# Runs the complete 5-phase training pipeline on DGX Spark.
#
# Phases:
#   Phase 0: Environment + data (no GPU needed)
#   Phase 1: SFT base model fine-tuning (8x A100)
#   Phase 2: Gate 1 - Retrieval (4x A100)
#   Phase 3: Gate 2 - FOL (8x A100)
#   Phase 4: Gate 3 - RL (4x A100)
#   Phase 5: Gate 4 - Verification (4x A100)
#   Phase 6: Merge gates + joint fine-tune
#
# Usage:
#   # Full pipeline (all 6 phases)
#   bash scripts/train_full_model.sh --phase all
#
#   # Single phase
#   bash scripts/train_full_model.sh --phase 1
#   bash scripts/train_full_model.sh --phase 2
#
#   # Smoke test first (recommended)
#   bash scripts/smoke_test.sh && bash scripts/train_full_model.sh
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_ROOT/checkpoints}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"

# Default: run all phases
PHASE="${1:-all}"
NUM_GPUS="${NUM_GPUS:-8}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

cd "$PROJECT_ROOT"

# Check environment
log "============================================================"
log "Distill-V4 Full Training Pipeline"
log "============================================================"
log "Project: $PROJECT_ROOT"
log "Checkpoints: $CHECKPOINT_DIR"
log "Phase: $PHASE"
log "GPUs: $NUM_GPUS"
log "============================================================"

# Detect GPUs
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    log "GPU count: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)"
else
    warn "nvidia-smi not found — CPU mode only"
fi

# =============================================================================
# PHASE 0: Data Generation (no GPU)
# =============================================================================
run_phase_0() {
    log ""
    log "============================================================"
    log "PHASE 0: Data Generation"
    log "============================================================"

    local DATA_OUTPUT="$DATA_DIR/raw/distillation_data.jsonl"

    if [ -f "$DATA_OUTPUT" ] && [ $(wc -l < "$DATA_OUTPUT") -gt 1000 ]; then
        log "Data already exists at $DATA_OUTPUT, skipping generation"
    else
        log "Generating distillation data (this may take a while)..."

        python "$SCRIPT_DIR/generate_data.py" \
            --mode generate \
            --output "$DATA_OUTPUT" \
            --num_samples 10000 \
            --parallel 8 || warn "API key not set, using template generation"

        # Filter English
        python "$SCRIPT_DIR/generate_data.py" \
            --mode filter \
            --input "$DATA_OUTPUT" \
            --output "$DATA_DIR/processed/english_data.jsonl"

        # Deduplicate
        python "$SCRIPT_DIR/generate_data.py" \
            --mode dedup \
            --input "$DATA_DIR/processed/english_data.jsonl" \
            --output "$DATA_DIR/processed/deduped_data.jsonl"

        # Split
        python "$SCRIPT_DIR/generate_data.py" \
            --mode split \
            --input "$DATA_DIR/processed/deduped_data.jsonl" \
            --output "$DATA_DIR/splits"
    fi

    log "✓ Phase 0 complete"
}

# =============================================================================
# PHASE 1: SFT Base Model
# =============================================================================
run_phase_1() {
    log ""
    log "============================================================"
    log "PHASE 1: SFT Base Model (Qwen2.5-Coder-7B)"
    log "============================================================"
    log "This will fine-tune the base model on your distillation data."
    log "Expected time on 8x A100 80GB: ~24-48 hours"
    log "============================================================"

    local OUTPUT="$CHECKPOINT_DIR/sft_base"

    # Check if already trained
    if [ -f "$OUTPUT/final/pytorch_model.bin" ] || [ -d "$OUTPUT/final" ]; then
        log "SFT base already trained, skipping Phase 1"
        return 0
    fi

    deepspeed --num_gpus=$NUM_GPUS \
        "$PROJECT_ROOT/src/training/train_sft_base.py" \
        --config "$PROJECT_ROOT/configs/sft_base.yaml" \
        --data_path "$DATA_DIR/splits/train.jsonl" \
        --output_dir "$OUTPUT" \
        2>&1 | tee "$CHECKPOINT_DIR/logs/sft_base.log"

    log "✓ Phase 1 complete"
}

# =============================================================================
# PHASE 2: Gate 1 — Retrieval
# =============================================================================
run_phase_2() {
    log ""
    log "============================================================"
    log "PHASE 2: Gate 1 — Knowledge Retrieval (2B)"
    log "============================================================"

    local OUTPUT="$CHECKPOINT_DIR/gate1_retrieval"

    if [ -f "$OUTPUT/final/pytorch_model.bin" ] || [ -d "$OUTPUT/final" ]; then
        log "Gate 1 already trained, skipping"
        return 0
    fi

    deepspeed --num_gpus=4 \
        "$PROJECT_ROOT/src/training/train_gate1_retrieval.py" \
        --config "$PROJECT_ROOT/configs/gate1_retrieval.yaml" \
        --data_path "$DATA_DIR/splits/train.jsonl" \
        --output_dir "$OUTPUT" \
        2>&1 | tee "$CHECKPOINT_DIR/logs/gate1.log"

    log "✓ Phase 2 complete"
}

# =============================================================================
# PHASE 3: Gate 2 — FOL
# =============================================================================
run_phase_3() {
    log ""
    log "============================================================"
    log "PHASE 3: Gate 2 — FOL Symbolic Reasoning (4B)"
    log "============================================================"

    local OUTPUT="$CHECKPOINT_DIR/gate2_fol"

    if [ -f "$OUTPUT/final/pytorch_model.bin" ] || [ -d "$OUTPUT/final" ]; then
        log "Gate 2 already trained, skipping"
        return 0
    fi

    deepspeed --num_gpus=8 \
        "$PROJECT_ROOT/src/training/train_gate2_fol.py" \
        --config "$PROJECT_ROOT/configs/gate2_fol.yaml" \
        --data_path "$DATA_DIR/splits/train.jsonl" \
        --output_dir "$OUTPUT" \
        2>&1 | tee "$CHECKPOINT_DIR/logs/gate2.log"

    log "✓ Phase 3 complete"
}

# =============================================================================
# PHASE 4: Gate 3 — RL
# =============================================================================
run_phase_4() {
    log ""
    log "============================================================"
    log "PHASE 4: Gate 3 — RL (GRPO) (1B)"
    log "============================================================"

    local OUTPUT="$CHECKPOINT_DIR/gate3_rl"

    if [ -f "$OUTPUT/final/pytorch_model.bin" ] || [ -d "$OUTPUT/final" ]; then
        log "Gate 3 already trained, skipping"
        return 0
    fi

    deepspeed --num_gpus=4 \
        "$PROJECT_ROOT/src/training/train_gate3_rl.py" \
        --config "$PROJECT_ROOT/configs/gate3_rl.yaml" \
        --data_path "$DATA_DIR/splits/train.jsonl" \
        --output_dir "$OUTPUT" \
        2>&1 | tee "$CHECKPOINT_DIR/logs/gate3.log"

    log "✓ Phase 4 complete"
}

# =============================================================================
# PHASE 5: Gate 4 — Verification
# =============================================================================
run_phase_5() {
    log ""
    log "============================================================"
    log "PHASE 5: Gate 4 — Verification (3B)"
    log "============================================================"

    local OUTPUT="$CHECKPOINT_DIR/gate4_verification"

    if [ -f "$OUTPUT/final/pytorch_model.bin" ] || [ -d "$OUTPUT/final" ]; then
        log "Gate 4 already trained, skipping"
        return 0
    fi

    deepspeed --num_gpus=4 \
        "$PROJECT_ROOT/src/training/train_gate4_verification.py" \
        --config "$PROJECT_ROOT/configs/gate4_verification.yaml" \
        --data_path "$DATA_DIR/splits/train.jsonl" \
        --output_dir "$OUTPUT" \
        2>&1 | tee "$CHECKPOINT_DIR/logs/gate4.log"

    log "✓ Phase 5 complete"
}

# =============================================================================
# PHASE 6: Merge Gates
# =============================================================================
run_phase_6() {
    log ""
    log "============================================================"
    log "PHASE 6: Merge Gates → Full 30B Model"
    log "============================================================"

    local OUTPUT="$CHECKPOINT_DIR/full_model_30b"

    python "$PROJECT_ROOT/src/training/merge_gates.py" \
        --base_model "$CHECKPOINT_DIR/sft_base/final" \
        --gate1 "$CHECKPOINT_DIR/gate1_retrieval/final" \
        --gate2 "$CHECKPOINT_DIR/gate2_fol/final" \
        --gate3 "$CHECKPOINT_DIR/gate3_rl/final" \
        --gate4 "$CHECKPOINT_DIR/gate4_verification/final" \
        --output "$OUTPUT" \
        --strategy sequential \
        2>&1 | tee "$CHECKPOINT_DIR/logs/merge.log"

    log "✓ Phase 6 complete"
}

# =============================================================================
# Main dispatcher
# =============================================================================
mkdir -p "$CHECKPOINT_DIR/logs"

case "$PHASE" in
    all|0)
        run_phase_0
        ;;
esac

case "$PHASE" in
    all|1)
        run_phase_1
        ;;
esac

case "$PHASE" in
    all|2)
        run_phase_2
        ;;
esac

case "$PHASE" in
    all|3)
        run_phase_3
        ;;
esac

case "$PHASE" in
    all|4)
        run_phase_4
        ;;
esac

case "$PHASE" in
    all|5)
        run_phase_5
        ;;
esac

case "$PHASE" in
    all|6)
        run_phase_6
        ;;
esac

log ""
log "============================================================"
log "✓ TRAINING PIPELINE COMPLETE"
log "============================================================"
log ""
log "Next steps:"
log "  1. Evaluate: python scripts/evaluate.py --model $CHECKPOINT_DIR/full_model_30b"
log "  2. Quantize: python src/inference/quantize.py --model $CHECKPOINT_DIR/full_model_30b"
log "  3. Deploy:  python src/inference/api_server.py --model $CHECKPOINT_DIR/full_model_30b-int8"
log ""
