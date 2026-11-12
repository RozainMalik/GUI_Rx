"""Zoom + full-view fusion (the paper's inference section): fuse the full-view map and the zoom map, then read out the click.
fuse_l<λ> is a fixed exponent; fuse_cv picks λ by two-fold cross-fitting.

  p_final ∝ p_full^λ · p_zoom on the zoom grid   (both max-normalised; λ = 0 is the zoom map alone)
  fuse_cv: λ ∈ LAMBDAS chosen by two-fold cross-fitting, never on the half it is scored on.
"""
from __future__ import annotations

import numpy as np

LAMBDAS = (0.25, 0.5, 1.0, 2.0)
SPLIT_SEED = 20261013                   


def fast_readout(scores, threshold=0.3):
    """GUI-AIMA's region readout (gui_aima.points.get_prediction_region_point, return_all_regions=False) on a (gh, gw)
    map: the 4-connected component of cells > threshold * max holding the argmax; its score-weighted cell-centre mean."""
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


def sample_on_zoom(full, zoom_shape, box, native_size):
    """Value of the full-view map (over the whole image, normalised coords) at each zoom-grid cell centre."""
    gh2, gw2 = zoom_shape
    x0, y0, x1, y1 = box
    W, H = native_size
    gh, gw = full.shape
    nx = (x0 + (np.arange(gw2) + 0.5) / gw2 * (x1 - x0)) / W
    ny = (y0 + (np.arange(gh2) + 0.5) / gh2 * (y1 - y0)) / H
    jx = np.clip(np.floor(nx * gw).astype(int), 0, gw - 1)
    jy = np.clip(np.floor(ny * gh).astype(int), 0, gh - 1)
    return full[np.ix_(jy, jx)]


def fused_point(full, zmap, box, native_size, lam):
    """Fused click, normalised to the native image. full: pass-1 map; zmap: zoom-pass map; box: zoom crop (native px)."""
    pf = sample_on_zoom(np.asarray(full), np.asarray(zmap).shape, box, native_size).astype(np.float64)
    pz = np.asarray(zmap, dtype=np.float64)
    pf, pz = pf / max(pf.max(), 1e-12), pz / max(pz.max(), 1e-12)
    p = fast_readout((np.maximum(pf, 0) ** lam) * pz if lam > 0 else pz)
    x0, y0, x1, y1 = box
    W, H = native_size
    return (x0 + p[0] * (x1 - x0)) / W, (y0 + p[1] * (y1 - y0)) / H


def cross_fit(sids, acc_of):
    """Two-fold cross-fitting: sids sorted, permuted with SPLIT_SEED, split in halves; on each half the λ with
    the highest acc_of(λ, half) (first in LAMBDAS order on ties) is applied to the other half.
    Returns ({sid: λ}, [λ chosen on fold A, λ chosen on fold B])."""
    sids = sorted(sids)
    n = len(sids)
    idx = np.random.default_rng(SPLIT_SEED).permutation(n)
    folds = (idx[: n // 2], idx[n // 2:])
    pick, chosen = {}, []
    for a, b in (folds, folds[::-1]):
        sub = [sids[i] for i in a]
        best = max(LAMBDAS, key=lambda lam: acc_of(lam, sub))
        chosen.append(best)
        for i in b:
            pick[sids[i]] = best
    return pick, chosen
