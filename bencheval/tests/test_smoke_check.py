"""scripts/smoke_check.py: a smoke step passes if at least one point and at least half of its predictions parsed."""
import json
import subprocess
import sys

import pytest

from benchev import ROOT


def _preds(root, points, d="1step_limit20"):
    p = root / "m" / "medspot" / d
    p.mkdir(parents=True)
    (p / "predictions.jsonl").write_text("".join(json.dumps({"sid": str(i), "point": pt}) + "\n"
                                                 for i, pt in enumerate(points)))


def _check(root, lim="20"):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "smoke_check.py"), str(root), "m", "medspot", "1step", lim],
                          capture_output=True, text=True)


@pytest.mark.parametrize("points,ok", [([[0.1, 0.2]] * 3, True), ([[0.1, 0.2], None], True),
                                       ([[0.1, 0.2], None, None], False), ([None] * 3, False)])
def test_threshold(tmp_path, points, ok):
    _preds(tmp_path, points)
    p = _check(tmp_path)
    assert (p.returncode == 0) == ok, p.stdout + p.stderr


def test_missing_dir_fails(tmp_path):
    assert _check(tmp_path).returncode != 0


def _zf(root, ref_root, shift_first=0.0, shift_zero=0.0, ref=True):
    """10 zf smoke records (largest2-style dir) and, optionally, the model's 1-step reference run."""
    d = root / "m" / "uivision" / "zf_largest10"
    d.mkdir(parents=True)
    recs, refs = [], []
    for i in range(10):
        pt = [0.3, 0.4]
        recs.append({"sid": str(i), "point": pt, "image_size": [1000, 1000], "first_point": [0.3 + shift_first, 0.4],
                     "zoom_point": pt, "fuse": {"0": [0.3 + shift_zero, 0.4], "1.0": pt}})
        refs.append({"sid": str(i), "point": [0.3, 0.4]})
    (d / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    if ref:
        r = ref_root / "m" / "uivision" / "1step"
        r.mkdir(parents=True)
        (r / "predictions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in refs))


def _zf_check(root, ref_root):
    import os
    env = dict(os.environ, ZF_REF=os.pathsep.join(["/nonexistent", str(ref_root)]))
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "smoke_check.py"), str(root), "m", "uivision", "zf",
                           "10"], capture_output=True, text=True, env=env)


@pytest.mark.parametrize("shift_first,shift_zero,ref,ok", [(0.0, 0.0, True, True), (0.002, 0.003, True, True),
                                                           (0.05, 0.0, True, False), (0.0, 0.05, True, False),
                                                           (0.0, 0.0, False, False)])
def test_zf_parity_gate(tmp_path, shift_first, shift_zero, ref, ok):
    """Pass 1 must reproduce the 1-step click and λ=0 the zoom click (4 px on 1000 px = 0.004); no reference = fail."""
    _zf(tmp_path / "res", tmp_path / "ref", shift_first, shift_zero, ref)
    p = _zf_check(tmp_path / "res", tmp_path / "ref")
    assert (p.returncode == 0) == ok, p.stdout + p.stderr
