#!/usr/bin/env bash
# =============================================================================
# VecFormer Environment Setup (vecformer_prod)
# Targets: Python 3.9 | PyTorch 2.5 | CUDA 12.1
# =============================================================================
set -euo pipefail

ENV_NAME="vecformer_prod"
PYTHON_VER="3.9"
PYTORCH_VER="2.5.0"
CUDA_TAG="cu121"                     # PyTorch CUDA toolkit tag
SPCONV_PKG="spconv-cu120"            # closest spconv build for CUDA 12.x
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ──────────────────────────────────────────────────────────────
# 0. Preflight checks
# ──────────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install Miniconda/Anaconda first." >&2
    exit 1
fi

echo "============================================="
echo "  VecFormer Environment Setup"
echo "  Env:    ${ENV_NAME}"
echo "  Python: ${PYTHON_VER}"
echo "  PyTorch: ${PYTORCH_VER} (${CUDA_TAG})"
echo "============================================="

# ──────────────────────────────────────────────────────────────
# 1. Create Conda environment
# ──────────────────────────────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[INFO] Conda env '${ENV_NAME}' already exists -- reusing it."
else
    echo "[STEP 1/6] Creating Conda environment '${ENV_NAME}' with Python ${PYTHON_VER} ..."
    conda create -y -n "${ENV_NAME}" python="${PYTHON_VER}"
fi

# Activate inside this script (works in both bash and zsh)
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
echo "[INFO] Active env: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "[INFO] Python: $(python --version)"

# ──────────────────────────────────────────────────────────────
# 2. Install PyTorch 2.5 + CUDA 12.1
# ──────────────────────────────────────────────────────────────
echo "[STEP 2/6] Installing PyTorch ${PYTORCH_VER} (${CUDA_TAG}) ..."
pip install torch=="${PYTORCH_VER}" torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

python -c "import torch; print(f'  -> torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}')"

# ──────────────────────────────────────────────────────────────
# 3. Install requirements.txt (swap spconv-cu118 -> cu120)
# ──────────────────────────────────────────────────────────────
echo "[STEP 3/6] Installing requirements.txt (with CUDA 12.x-compatible spconv) ..."
# Install everything except the CUDA-11.8 spconv pin
pip install $(grep -v '^spconv' "${REPO_DIR}/requirements.txt" | tr '\n' ' ')

# Install spconv built for CUDA 12.x instead
pip install "${SPCONV_PKG}"

# ──────────────────────────────────────────────────────────────
# 4. Install torch-scatter (matches PyTorch + CUDA version)
# ──────────────────────────────────────────────────────────────
echo "[STEP 4/6] Installing torch-scatter ..."
pip install torch-scatter -f "https://data.pyg.org/whl/torch-${PYTORCH_VER}+${CUDA_TAG}.html"

# ──────────────────────────────────────────────────────────────
# 5. Install Flash-Attention, PyMuPDF, Shapely
# ──────────────────────────────────────────────────────────────
echo "[STEP 5/6] Installing Flash-Attention (builds from source -- may take a few minutes) ..."
pip install flash-attn --no-build-isolation

echo "[STEP 5/6] Installing PyMuPDF and Shapely ..."
pip install PyMuPDF shapely

# ──────────────────────────────────────────────────────────────
# 6. PTv3 custom CUDA kernels -- NOT APPLICABLE
# ──────────────────────────────────────────────────────────────
echo "[STEP 6/6] PTv3 custom CUDA kernels ..."
echo ""
echo "  NOTE: This repository contains NO custom CUDA kernels."
echo "  The PTv3 backbone relies on pre-built external packages:"
echo "    - spconv     (sparse convolution)       -> installed via ${SPCONV_PKG}"
echo "    - torch-scatter (scatter/segment ops)   -> installed above"
echo "    - flash-attn  (Flash Attention)          -> installed above"
echo ""
echo "  There is no kernels/ directory or .cu/.cpp source to compile."
echo "  All GPU-accelerated ops come from these pip-installed libraries."
echo ""

# ──────────────────────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────────────────────
echo "============================================="
echo "  Verifying installation"
echo "============================================="

python -c "
import sys
print(f'Python      : {sys.version}')

import torch
print(f'PyTorch     : {torch.__version__}')
print(f'CUDA avail  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU         : {torch.cuda.get_device_name(0)}')

import spconv
print(f'spconv      : {spconv.__version__}')

import torch_scatter
print(f'torch_scatter: OK')

try:
    import flash_attn
    print(f'flash_attn  : {flash_attn.__version__}')
except ImportError:
    print(f'flash_attn  : NOT INSTALLED (optional, CPU-only fallback exists)')

import fitz
print(f'PyMuPDF     : {fitz.version_bind}')

import shapely
print(f'Shapely     : {shapely.__version__}')

import transformers
print(f'transformers: {transformers.__version__}')

print()
print('All dependencies verified successfully.')
"

echo ""
echo "============================================="
echo "  Setup complete!"
echo "  Activate with:  conda activate ${ENV_NAME}"
echo "============================================="
