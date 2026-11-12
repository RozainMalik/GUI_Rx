"""MedSPOT metrics under both hit rules used for MedSPOT models.

strict  = medspot_strict.core unchanged (point inside the GT box).
tol14px = the legacy evaluators' rule (GUI-Actor, OS-Atlas, Aguvis, RegionFocus, ...): a 28x28 px patch centred on the
          point overlaps the GT box, in original-image pixels, edges inclusive.
Both use core's step selection (every step with a bbox action, no early stop) and core's compute_metrics: a point the
patch rule accepts is moved onto the nearest point of the box, and core scores the result. The metric definitions are
therefore identical and only the hit test differs.
"""
from medspot_strict.core import compute_metrics

PATCH_PX = 14


def tol_hit(point, bbox, size, px=PATCH_PX):
    if point is None:
        return False
    w, h = size
    x, y = point[0] * w, point[1] * h
    x1, y1, x2, y2 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h
    return x + px >= x1 and x - px <= x2 and y + px >= y1 and y - px <= y2


def tol_records(records, px=PATCH_PX):
    out = {}
    for k, r in records.items():
        p = r.get("pred_point")
        if p is not None and tol_hit(p, r["gt_bbox"], r["image_size"], px):
            x1, y1, x2, y2 = r["gt_bbox"]
            r = {**r, "pred_point": [min(max(p[0], x1), x2), min(max(p[1], y1), y2)]}
        out[k] = r
    return out


def both_rules(tasks, records):
    return {"strict": compute_metrics(tasks, records), "tol14px": compute_metrics(tasks, tol_records(records))}
