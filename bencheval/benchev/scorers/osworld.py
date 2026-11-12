"""OSWorld-G. acc = official semantics (bbox / polygon / refusal); acc_upstream_compat = GUI-AIMA's
osworld-g_two_stage.py (box_coordinates[:4] read as x,y,w,h for every sample, so refusals [0,0,0,0] always miss);
acc_bbox470 = bbox samples only."""
from __future__ import annotations

from benchev.geometry import point_in_box, point_in_polygon
from benchev.scorers import pointbox


def official_hit(s, p):
    if s.gt_kind == "refusal":
        return p is None
    if s.gt_kind == "polygon":
        return point_in_polygon(p, s.gt)
    return point_in_box(p, s.gt)


def upstream_hit(s, p):
    if p is None:
        return False
    x, y, w, h = s.tags["raw_box"][:4]
    W, H = s.size
    return point_in_box(p, (x / W, y / H, (x + w) / W, (y + h) / H))


def score(samples, preds, split_keys):
    m = pointbox.score(samples, preds, split_keys, official_hit)
    m["acc_upstream_compat"] = pointbox.score(samples, preds, [], upstream_hit)["acc"]
    b = pointbox.score([s for s in samples if s.gt_kind == "box"], preds, [])
    m["acc_bbox470"], m["n_bbox470"] = b["acc"], b["n"]
    return m
