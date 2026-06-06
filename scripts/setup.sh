#!/bin/bash
# =============================================================================
# Distill-V4 Setup Script
# =============================================================================
# Sets up the complete environment for DGX Spark or local GPU training.
#
# Usage:
#   bash scripts/setup.sh              # Local GPU
#   bash scripts/setup.sh --dgx        # DGX Spark
#   bash scripts/setup.sh --check      # Verify only, no install
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODE="${1:-local}"

echo "============================================================"
echo "Distill-V4 Setup"
echo "============================================================"
echo "Mode: $MODE"
echo "Project root: $PROJECT_ROOT"
echo "Python: $(which python || echo 'not found')"
echo "============================================================"

# =============================================================================
# Step 1: Check CUDA/GPU
# =============================================================================
echo ""
echo "[1/5] Checking GPU..."

if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_COUNT=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1 | xargs)
    echo "  ✓ GPU: $GPU_NAME x $GPU_COUNT ($GPU_MEM)"
else
    echo "  ⚠ nvidia-smi not found — will run in CPU-only mode"
fi

# Check CUDA
if python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')" 2>/dev/null; then
    :
else
    echo "  ⚠ PyTorch not installed or CUDA not available"
fi

# =============================================================================
# Step 2: Create Python environment
# =============================================================================
echo ""
echo "[2/5] Creating Python environment..."

if [ -d "$PROJECT_ROOT/.venv" ]; then
    echo "  ✓ Virtualenv already exists at .venv"
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
else
    echo "  Creating virtualenv at .venv..."
    python3.11 -m venv "$PROJECT_ROOT/.venv"
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
    echo "  ✓ Virtualenv created"
fi

# Upgrade pip
"$PYTHON" -m pip install --upgrade pip wheel setuptools

# =============================================================================
# Step 3: Install PyTorch
# =============================================================================
echo ""
echo "[3/5] Installing PyTorch..."

# Detect CUDA version
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "none")

if [ "$MODE" == "dgx" ]; then
    # DGX Spark: use pre-installed PyTorch via conda
    echo "  DGX mode: expecting system PyTorch"
elif command -v nvidia-smi &> /dev/null; then
    # Local GPU: install PyTorch with CUDA support
    "$PYTHON" -m pip install \
        torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
        --index-url https://download.pytorch.org/whl/cu121
    echo "  ✓ PyTorch installed (CUDA $CUDA_VERSION)"
else
    # CPU only
    "$PYTHON" -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
    echo "  ✓ PyTorch installed (CPU only)"
fi

# =============================================================================
# Step 4: Install Python dependencies
# =============================================================================
echo ""
echo "[4/5] Installing Python dependencies..."

"$PYTHON" -m pip install \
    -r "$PROJECT_ROOT/requirements.txt" \
    --quiet

echo "  ✓ Dependencies installed"

# =============================================================================
# Step 5: Verify installation
# =============================================================================
echo ""
echo "[5/5] Verifying installation..."

"$PYTHON" -c "
import sys
pkgs = ['torch', 'transformers', 'deepspeed', 'accelerate', 'wandb', 'numpy', 'pyyaml']
for pkg in pkgs:
    try:
        m = __import__(pkg)
        v = getattr(m, '__version__', 'unknown')
        print(f'  ✓ {pkg} {v}')
    except ImportError:
        print(f'  ✗ {pkg} NOT INSTALLED')
        sys.exit(1)
" || { echo "  ✗ Verification failed"; exit 1; }

# =============================================================================
# Create directories
# =============================================================================
echo ""
echo "[Done] Creating project directories..."
mkdir -p "$PROJECT_ROOT"/{checkpoints,data/{raw,processed,splits},logs}
echo "  ✓ Directories created"

echo ""
echo "============================================================"
echo "✓ SETUP COMPLETE"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Generate data:"
echo "     python scripts/generate_data.py --mode generate --num_samples 10000"
echo ""
echo "  2. Run smoke test:"
echo "     bash scripts/smoke_test.sh"
echo ""
echo "  3. Start full training:"
echo "     bash scripts/train_full_model.sh"
echo ""