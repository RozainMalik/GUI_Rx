#!/usr/bin/env bash
# Create .venv with the pinned training stack (CUDA 12.6 wheels). Needs Python 3.11, nvcc and a C++20 compiler
# (gcc >= 10) for causal-conv1d. Set TORCH_CUDA_ARCH_LIST to your GPU (9.0 = H100, 8.6 = A10, 8.9 = L4).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || python3.11 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip wheel setuptools ninja packaging
pip install -q torch==2.14.0+cu126 torchvision==0.29.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install -q -r <(grep -v -E "^(#|torch)" requirements.txt)
pip install -q flash-linear-attention
export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc)")")} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
pip install -q --no-build-isolation causal-conv1d
python -c "import torch, transformers, fla, causal_conv1d; print('OK torch', torch.__version__, 'transformers', transformers.__version__)"
