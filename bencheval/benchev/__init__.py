"""BenchEval: unified GUI-grounding evaluation (see README.md)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_VENDOR = str(ROOT / "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
