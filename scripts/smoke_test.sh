#!/bin/bash
# =============================================================================
# Distill-V4 Smoke Test Script
# =============================================================================
# Runs smoke tests for all 4 gates and the full sequential pass.
# Use this to verify the architecture works BEFORE submitting a full training job.
#
# Usage:
#   bash scripts/smoke_test.sh
#   bash scripts/smoke_test.sh --device cuda
# =============================================================================

set -e  # Exit on first error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "============================================================"
echo "Distill-V4 Smoke Test"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    echo "GPU memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 | xargs) per GPU"
fi
echo "============================================================"

# Parse args
DEVICE="${1:-auto}"

# Determine device
if [ "$DEVICE" == "cuda" ] || [ "$DEVICE" == "cpu" ]; then
    DEVICE_ARG="--device $DEVICE"
else
    DEVICE_ARG=""  # Auto-detect
fi

# =============================================================================
# Test 1: Python syntax and import check
# =============================================================================
echo ""
echo "[Test 1] Python syntax and import check..."
python -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from src.models.distill_v4_model import (
    KnowledgeRetrievalGate,
    SymbolicReasoningGate,
    RLGate,
    VerificationGate,
    count_parameters,
)
print('  ✓ All imports successful')
" || { echo "  ✗ Import failed"; exit 1; }

# =============================================================================
# Test 2: Gate parameter counts
# =============================================================================
echo ""
echo "[Test 2] Gate parameter counts..."
python -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from src.models.distill_v4_model import (
    KnowledgeRetrievalGate,
    SymbolicReasoningGate,
    RLGate,
    VerificationGate,
    count_parameters,
)

gates = [
    ('Retrieval', KnowledgeRetrievalGate(hidden_dim=4096, memory_size=5000)),
    ('FOL', SymbolicReasoningGate(hidden_dim=4096, intermediate_dim=2048, num_reasoning_steps=2)),
    ('RL', RLGate(hidden_dim=4096)),
    ('Verification', VerificationGate(hidden_dim=4096)),
]

total = 0
for name, gate in gates:
    params = count_parameters(gate)
    total += params
    print(f'  {name}: {params:,} params')
print(f'  TOTAL: {total:,} params ({total/1e9:.3f}B)')
" || { echo "  ✗ Parameter count failed"; exit 1; }

# =============================================================================
# Test 3: Forward pass on CPU
# =============================================================================
echo ""
echo "[Test 3] Forward pass on CPU..."
python "$SCRIPT_DIR/smoke_test.py" --device cpu 2>&1 | tee /tmp/smoke_test.log

if grep -q "PASSED" /tmp/smoke_test.log; then
    echo ""
    echo "============================================================"
    echo "✓ ALL SMOKE TESTS PASSED"
    echo "============================================================"
    echo ""
    echo "Architecture is verified. Ready to run full training:"
    echo "  bash scripts/train_full_model.sh"
    exit 0
else
    echo ""
    echo "============================================================"
    echo "✗ SMOKE TESTS FAILED — fix errors before training"
    echo "============================================================"
    exit 1
fi
