"""Zoom + fusion mode ("zf"): fuse_l<λ> / fuse_cv against a reference implementation."""
import json

import numpy as np
import pytest
from PIL import Image

from benchev.adapters.fake import FakeAdapter
from benchev.config import load_protocol
from benchev.fusion import LAMBDAS, SPLIT_SEED, cross_fit, fast_readout, fused_point
from benchev.geometry import area_crop_px
from benchev.inference import predict_sample
from benchev.run import run


# --- oracle: an independent reference implementation of the fusion functions ---------------------------------------

def _ref_fast_readout(scores, threshold=0.3):
    s = np.asarray(scores, dtype=np.float64)
    gh, gw = s.shape
    mask = s > s.max() * threshold
    start = np.unravel_index(int(np.argmax(s)), s.shape)
    seen = np.zeros_like(mask)
    seen[start] = True
    stack, cells = [start], []
    while stack:
        y, x = stack.pop()
        cells.append((y, x))
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < gh and 0 <= nx < gw and mask[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                stack.append((ny, nx))
    ys, xs = np.array(cells).T
    w = s[ys, xs]
    return float(((xs + 0.5) / gw * w).sum() / w.sum()), float(((ys + 0.5) / gh * w).sum() / w.sum())


def _ref_fused_point(full, zmap, row, lam):
    gw2, gh2 = row["zoom"]["grid"]
    x0, y0, x1, y1 = row["zoom"]["box"]
    W, H = row["native_size"]
    gh, gw = full.shape
    nx = (x0 + (np.arange(gw2) + 0.5) / gw2 * (x1 - x0)) / W
    ny = (y0 + (np.arange(gh2) + 0.5) / gh2 * (y1 - y0)) / H
    jx = np.clip(np.floor(nx * gw).astype(int), 0, gw - 1)
    jy = np.clip(np.floor(ny * gh).astype(int), 0, gh - 1)
    pf = full[np.ix_(jy, jx)].astype(np.float64)
    pz = zmap.astype(np.float64)
    pf, pz = pf / max(pf.max(), 1e-12), pz / max(pz.max(), 1e-12)
    p = _ref_fast_readout((np.maximum(pf, 0) ** lam) * pz if lam > 0 else pz)
    return (x0 + p[0] * (x1 - x0)) / W, (y0 + p[1] * (y1 - y0)) / H


def test_fast_readout_matches_reference():
    rng = np.random.default_rng(0)
    for _ in range(50):
        m = rng.random((17, 29)) ** 4
        assert fast_readout(m) == _ref_fast_readout(m)


@pytest.mark.parametrize("lam", [0, 0.25, 0.5, 1.0, 2.0])
def test_fused_point_matches_reference(lam):
    rng = np.random.default_rng(1)
    full, zmap = rng.random((34, 60)) ** 3, rng.random((38, 38)) ** 3
    box, size = (300, 120, 916, 736), (1920, 1080)
    row = {"zoom": {"grid": [38, 38], "box": list(box)}, "native_size": list(size)}
    assert fused_point(full, zmap, box, size, lam) == pytest.approx(_ref_fused_point(full, zmap, row, lam), abs=1e-12)


def test_cross_fit_matches_ref_rule():
    """Reference: rows sorted by id, permutation(SPLIT_SEED), halves; λ chosen on one half (first max in LAMBDAS order),
    applied to the other."""
    sids = [f"s{i:03d}" for i in range(40)]
    hits = {lam: {s: (int(s[1:]) % (k + 2) == 0) for s in sids} for k, lam in enumerate(LAMBDAS)}
    idx = np.random.default_rng(SPLIT_SEED).permutation(len(sids))
    folds = (idx[: len(sids) // 2], idx[len(sids) // 2:])
    want, chosen = {}, []
    for a, b in (folds, folds[::-1]):
        best = max(LAMBDAS, key=lambda l: sum(hits[l][sids[i]] for i in a))
        chosen.append(best)
        for i in b:
            want[sids[i]] = best
    got, got_chosen = cross_fit(sids, lambda lam, subset: 100.0 * sum(hits[lam][s] for s in subset) / len(subset))
    assert got == want and got_chosen == chosen


def _mapped(point, shape=(8, 8)):
    """A map whose GUI-AIMA readout is `point` (one hot cell)."""
    gh, gw = shape
    m = np.zeros(shape)
    m[min(gh - 1, int(point[1] * gh)), min(gw - 1, int(point[0] * gw))] = 1.0
    return m


def test_zf_first_pass_is_one_step_frame_and_zoom_is_two_step_crop():
    """UI-Vision: pass 1 = the 1-step frame (3200x1800 resize, 1-step max_pixels); pass 2 = the 2-step crop of the
    NATIVE image around pass 1's point, zoomed, with the 2-step max_pixels."""
    c = 4.5 / 8                                      # centre of cell 4 of 8: the fake map's readout is the fake click
    a = FakeAdapter(lambda img, ins: (c, c), crop_multiple=32, map_fn=lambda img, ins: _mapped((c, c)))
    p = load_protocol("uivision@v1")
    out = predict_sample(a, Image.new("RGB", (1920, 1080)), "x", p, "zf")
    assert a.calls[0]["size"] == (3200, 1800) and a.calls[0]["max_pixels"] == p.max_pixels["1step"]
    side = area_crop_px(1920, 1080, 616 ** 2, 5_760_000, 32)
    x0, y0, x1, y1 = out["crop_box"]
    assert (x1 - x0, y1 - y0) == (side, side)
    assert a.calls[1]["size"] == (int(side * p.zoom),) * 2 and a.calls[1]["max_pixels"] == p.max_pixels["2step"]
    assert out["first_point"] == (c, c) and out["input_size"] == (3200, 1800)
    assert set(out["fuse"]) == {str(l) for l in (0,) + LAMBDAS} and out["point"] == tuple(out["fuse"]["1.0"])
    # λ = 0 is the zoom map alone: its readout mapped back is the zoom click
    assert tuple(out["fuse"]["0"]) == pytest.approx(out["zoom_point"], abs=1e-9)


def test_zf_none_first_point():
    a = FakeAdapter(lambda img, ins: None, map_fn=lambda img, ins: None)
    out = predict_sample(a, Image.new("RGB", (800, 600)), "x", load_protocol("ssv2@v1"), "zf")
    assert out["point"] is None and out["crop_box"] is None and out["fuse"] is None and len(a.calls) == 1


def _ssv2(root, n=6):
    rows = []
    for i in range(n):
        Image.new("RGB", (100, 100)).save(root / f"{i}.png")
        rows.append({"img_filename": f"{i}.png", "bbox": [0, 0, 50, 50], "instruction": "x", "split": "web",
                     "data_type": "icon"})
    json.dump(rows, open(root / "subset.json", "w"))


def test_zf_run_scores_cross_fitted_and_per_lambda(tmp_path):
    d = tmp_path / "data" / "screenspot_v2_full"
    d.mkdir(parents=True)
    _ssv2(d)
    a = FakeAdapter(lambda i, t: (.2, .2), crop_multiple=32, map_fn=lambda i, t: _mapped((.2, .2)))
    s = run("ours-2b-ab-clin", "ssv2", "zf", tmp_path / "data", tmp_path / "raw", adapter=a)
    m = s["metrics"]
    assert s["mode"] == "zf" and m["acc"] == 100.0 and m["acc_zoom"] == 100.0
    assert all(m[f"acc_l{l}"] == 100.0 for l in LAMBDAS) and len(m["fuse_cv_lambdas"]) == 2
    rec = json.loads((tmp_path / "raw/ours-2b-ab-clin/ssv2/zf/predictions.jsonl").read_text().splitlines()[0])
    assert {"fuse", "zoom_point", "first_point", "crop_box"} <= set(rec)


def test_zf_rejects_medspot():
    a = FakeAdapter(lambda i, t: (.5, .5), map_fn=lambda i, t: _mapped((.5, .5)))
    with pytest.raises(NotImplementedError):
        run("ours-2b-ab-clin", "medspot", "zf", "nowhere", "nowhere", adapter=a)
