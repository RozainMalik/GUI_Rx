import os
import sys

# transformers binds the Gated-DeltaNet kernels (fla, causal-conv1d) at import time if the packages are installed,
# regardless of device, and the Triton kernels reject CPU tensors. QWEN35_TORCH_KERNELS=1 forces the PyTorch
# reference path (CPU tests on a machine that has fla). Must take effect before transformers' qwen3_5 modeling loads.
if os.environ.get("QWEN35_TORCH_KERNELS") == "1":
    for _pkg in ("fla", "causal_conv1d"):
        sys.modules.setdefault(_pkg, None)
