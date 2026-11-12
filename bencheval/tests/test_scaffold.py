import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "training" / "src"


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.mark.skipif(not SRC.exists(), reason="training/ not next to bencheval/")
def test_vendored_core_is_byte_identical():
    assert _sha(ROOT / "vendor/medspot_strict/core.py") == _sha(SRC / "medspot_strict/core.py")


def test_vendored_metrics_imports_core():
    import medspot_metrics
    assert medspot_metrics.PATCH_PX == 14


def test_package_imports_without_torch():
    # fresh interpreter: other tests may import torch into this process
    import subprocess
    import sys
    code = "import sys, benchev; assert 'torch' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT).returncode == 0
