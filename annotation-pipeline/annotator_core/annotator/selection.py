"""Selection: agreement against the four independent boxes, the consensus cluster,
tiers, the per-candidate `final` block and the per-frame choice. Pure Python over the record
dicts. (Not named select.py, so it never shadows stdlib select.)"""
from __future__ import annotations

from .geometry import agree, area, clusters, median_box, px_to_pct_xywh

INDEPENDENT = ("describe", "functional", "visual", "spatial")
VERDICT_RANK = {"correct": 0, "loose": 1, "partial": 2, "wrong_element": 3}
TIER_RANK = {"A": 0, "B": 1, "C": 2}


def independent_boxes(cand: dict) -> dict:
    """The S2 box and the three blind groundings that exist and pass sanity, keyed by name."""
    out = {}
    d = cand.get("describe") or {}
    if d.get("bbox_px") and d.get("box_sane") is None:
        out["describe"] = tuple(d["bbox_px"])
    g = cand.get("ground") or {}
    for v in INDEPENDENT[1:]:
        gv = g.get(v) or {}
        if gv.get("bbox_px") and gv.get("box_sane") is None:
            out[v] = tuple(gv["bbox_px"])
    return out


def agreement(final_box, independent: dict) -> float:
    """Fraction of the four independent boxes agreeing with the final box - always over 4, so
    a missing or insane box counts as disagreement."""
    return sum(1 for b in independent.values() if agree(tuple(final_box), b)) / 4


def pick_cluster(boxes: dict, prefer: str) -> list:
    """Largest agreement cluster; ties -> the one holding `prefer`, then the smaller median box."""
    comps = clusters(boxes)
    best = len(comps[0])
    tied = [c for c in comps if len(c) == best]
    if len(tied) > 1:
        with_pref = [c for c in tied if prefer in c]
        if with_pref:
            tied = with_pref
        tied.sort(key=lambda c: area(median_box(boxes[k] for k in c)))
    return tied[0]


def consensus(independent: dict):
    """(consensus box, cluster keys), or None when no independent box exists."""
    if not independent:
        return None
    cluster = pick_cluster(independent, prefer="describe")
    return median_box(independent[k] for k in cluster), cluster


def tier(verified, agreement_value, instructions_valid) -> str:
    if verified and instructions_valid and agreement_value >= 0.75:
        return "A"
    if verified and instructions_valid and agreement_value == 0.5:
        return "B"
    return "C"


def provisional_tier(cand: dict) -> str:
    """The tier as it stands after S4 - decides whether S5 runs."""
    cons, ver, d = cand.get("consensus"), cand.get("verify"), cand.get("describe") or {}
    if not cons or not ver:
        return "C"
    return tier(bool(ver.get("verified")), cons["agreement"], not d.get("instruction_errors"))


def candidate_calls(cand: dict) -> int:
    n = (cand.get("describe") or {}).get("calls", 0)
    for gv in (cand.get("ground") or {}).values():
        n += (gv or {}).get("calls", 0)
    n += (cand.get("verify") or {}).get("calls", 0)
    n += (cand.get("refine") or {}).get("calls", 0)
    return n


def finalize(cand: dict, width, height):
    """cand['final'] from the latest blocks: refine's box and re-verify when refine produced a
    box, else the consensus box and S4. None when the candidate never reached a box."""
    d = cand.get("describe")
    if not d or not d.get("instructions"):
        return None
    ref = cand.get("refine") or {}
    if ref.get("final_bbox_px"):
        box = tuple(ref["final_bbox_px"])
        ver = ref.get("reverify") or {}
        rewrite = ref.get("rewrite") or {}
        if rewrite.get("instructions"):
            instructions, errors = rewrite["instructions"], rewrite.get("instruction_errors", [])
        else:
            instructions, errors = d["instructions"], d.get("instruction_errors", [])
        changed = ref.get("instructions_changed", [])
    else:
        cons = cand.get("consensus")
        if not cons:
            return None
        box = tuple(cons["bbox_px"])
        ver = cand.get("verify") or {}
        instructions, errors, changed = d["instructions"], d.get("instruction_errors", []), []
    agr = agreement(box, independent_boxes(cand))
    verified = bool(ver.get("verified"))
    return {"bbox_px": list(box), "bbox_pct": px_to_pct_xywh(box, width, height), "instructions": instructions,
            "instructions_changed": changed, "verified": verified, "verdict": ver.get("verdict"),
            "readback_match": ver.get("readback_match"), "agreement": agr,
            "tier": tier(verified, agr, not errors), "calls": candidate_calls(cand)}


def rank_key(cand: dict) -> tuple:
    """Order: tier A < B < C, agreement desc, verdict rank, state != plain first, proposer rank."""
    f = cand["final"]
    return (TIER_RANK[f["tier"]], -f["agreement"], VERDICT_RANK.get(f["verdict"], 4),
            0 if cand["element"]["state"] != "plain" else 1, cand["rank"])


def choose(candidates: list):
    eligible = [i for i, c in enumerate(candidates) if c.get("status") == "ok" and c.get("final")]
    if not eligible:
        return None
    return min(eligible, key=lambda i: rank_key(candidates[i]))


def frame_status(record: dict) -> str:
    if (record.get("propose") or {}).get("rejected_reason"):
        return "abstained"
    sel = record.get("selected")
    if sel is None:
        statuses = {c.get("status") for c in record.get("candidates", [])}
        return "unreadable" if "unreadable" in statuses else "abstained"
    return "shipped" if record["candidates"][sel]["final"]["tier"] in ("A", "B") else "lowconf"
