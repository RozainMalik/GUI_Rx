from __future__ import annotations

from collections import defaultdict
from itertools import product

from benchev.geometry import point_in_box


def _pct(h, t):
    return round(100.0 * h / t, 2) if t else 0.0


def split_names(sample, keys):
    """One name per split the sample is in; a list-valued tag (OSWorld-G category) puts it in each listed group."""
    vals = [v if isinstance(v := sample.tags.get(k), (list, tuple)) else [v] for k in keys]
    return ["/".join(f"{k}={v}" for k, v in zip(keys, combo)) for combo in product(*vals)]


def score(samples, preds, split_keys, hit_fn=None):
    hit_fn = hit_fn or (lambda s, p: point_in_box(p, s.gt))
    hit = tot = missing = 0
    sh, st = defaultdict(int), defaultdict(int)
    for s in samples:
        if s.sid not in preds:
            missing += 1
            h = 0                       # never ask hit_fn: an OSWorld-G refusal would count a missing point as a hit
        else:
            h = hit_fn(s, preds[s.sid])
        hit += h
        tot += 1
        for keys in split_keys:
            for name in split_names(s, keys):
                sh[name] += h
                st[name] += 1
    return {"acc": _pct(hit, tot), "n": tot, "n_missing": missing,
            "split": {k: _pct(sh[k], st[k]) for k in sorted(st)}, "split_n": dict(sorted(st.items()))}
