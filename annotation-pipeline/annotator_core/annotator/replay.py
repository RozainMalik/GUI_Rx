"""reset --parse-only: re-interpret one stage's stored raw replies with the current
parser and validators, making no model call. Because temperature is 0 and every raw reply is
kept, a parser fix never needs the GPU. A candidate whose result changed has its later blocks
cleared; when anything changed, stages_done is truncated to the stage so `annotate` reruns the
later stages - skipping candidates whose blocks survived."""
from __future__ import annotations

import copy
import json

from .contract import ContractError
from .geometry import CoordSpace, full_space
from .records import CANDIDATE_BLOCKS, STAGES, candidate_from_seed, set_candidate_status
from .stages import describe as s2
from .stages import ground as s3
from .stages import propose as s1
from .stages import verify as s4
from .validators import VARIANTS, SoftValidationError, ValidationError, box_sanity, readback_match

REPLAYABLE = ("propose", "describe", "ground", "verify")


def _same(a, b) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _try(interpret, raw):
    """(value, error, soft_errors) - the same three outcomes call_validated produces."""
    try:
        return interpret(raw), None, []
    except SoftValidationError as soft:
        return soft.value, None, list(soft.errors)
    except (ContractError, ValidationError) as e:
        return None, str(e), []


def _clear_after(cand, stage):
    later = STAGES[STAGES.index(stage) + 1:]
    for s in later:
        for block in CANDIDATE_BLOCKS.get(s, ()):
            cand[block] = None
    if cand.get("status_stage") in later:
        set_candidate_status(cand, "ok", None, None)


def _reparse_propose(record, width, height, mode) -> int:
    p = record.get("propose")
    if not p or not p.get("raw"):
        return 0
    value, error, _ = _try(lambda raw: s1.interpret(raw, full_space(width, height, mode)), p["raw"])
    new_p = dict(p, intent=value["intent"] if value else None,
                 focus_region=value["focus_region"] if value else None,
                 rejected_reason=error if value is None else (None if value["candidates"] else "no_candidates"))
    new_cands = [candidate_from_seed(s) for s in value["candidates"]] if value else []
    seed_key = lambda c: json.dumps([c["element"], c["seed_bbox_px"]], sort_keys=True)
    old_keys = [seed_key(c) for c in record["candidates"]]
    new_keys = [seed_key(c) for c in new_cands]
    if _same(new_p, p) and old_keys == new_keys:
        return 0
    record["propose"] = new_p
    if new_p["rejected_reason"]:
        record["candidates"], record["stages_done"], record["status"] = [], list(STAGES), "abstained"
    else:
        kept = dict(zip(old_keys, record["candidates"]))
        record["candidates"] = [kept.get(k, c) for k, c in zip(new_keys, new_cands)]
        record["stages_done"], record["status"] = ["propose"], None
    record["selected"] = None
    return 1


def _reparse_describe(cand, width, height, mode) -> bool:
    d = cand.get("describe")
    if not d or not d.get("raw"):
        return False
    seed_box, anchor = tuple(cand["seed_bbox_px"]), cand["element"]["text_on_screen"]
    space = CoordSpace(tuple(d["window"]), tuple(d["shown_size"]), mode)
    value, error, soft = _try(lambda raw: s2.interpret(raw, space, seed_box, anchor), d["raw"])
    new = dict(d, error=error, instruction_errors=soft)
    if value is None or not value["present"]:
        new.update(present=value is None, bbox_1000=None, bbox_px=None, instructions=None, box_sane=None)
    else:
        new.update(value)
        if d.get("truncated"):
            new["bbox_px"], new["bbox_1000"] = list(seed_box), None
        new["box_sane"] = box_sanity(tuple(new["bbox_px"]), width, height)
    if _same(new, d):
        return False
    cand["describe"] = new
    if error:
        set_candidate_status(cand, "unreadable", "describe", error)
    elif not new["present"]:
        set_candidate_status(cand, "dropped", "describe", "not_in_region")
    elif cand.get("status_stage") == "describe":
        set_candidate_status(cand, "ok", None, None)
    return True


def _reparse_ground(cand, width, height, mode) -> bool:
    g = cand.get("ground")
    if not g:
        return False
    full = full_space(width, height, mode)
    new = {}
    for v in VARIANTS:
        gv = g.get(v) or {}
        if not gv.get("raw"):
            new[v] = gv
            continue
        value, error, _ = _try(lambda raw: s3.interpret(raw, full, allow_empty=False), gv["raw"])
        if value is None:
            new[v] = dict(gv, bbox_1000=None, bbox_px=None, box_sane=None, label="", error=error)
        else:
            new[v] = dict(gv, bbox_1000=value["bbox_1000"], bbox_px=value["bbox_px"],
                          box_sane=box_sanity(tuple(value["bbox_px"]), width, height), label=value["label"], error=None)
    if _same(new, g):
        return False
    cand["ground"] = new
    return True


def _reparse_verify(cand, width, height, mode) -> bool:
    ver = cand.get("verify")
    if not ver or not ver.get("raw"):
        return False
    space = CoordSpace(tuple(ver["window"]), tuple(ver["shown_size"]), mode)
    value, error, _ = _try(lambda raw: s4.interpret(raw, space), ver["raw"])
    new = dict(ver, error=error)
    if value is None:
        new.update(text_in_box="", element_in_box="", verdict=None, reason="", readback_match=None,
                   corrected_bbox_1000=None, corrected_bbox_px=None, corrected_box_sane=None, verified=False)
    else:
        new.update(value)
        cpx = value["corrected_bbox_px"]
        new["corrected_box_sane"] = box_sanity(tuple(cpx), width, height) if cpx else None
        new["readback_match"] = readback_match(cand["describe"]["text_on_screen"], value["text_in_box"])
        new["verified"] = value["verdict"] == "correct" and new["readback_match"] in (True, None)
    if _same(new, ver):
        return False
    cand["verify"] = new
    return True


def reparse(record, stage) -> int:
    if stage not in REPLAYABLE:
        raise ValueError(f"--parse-only supports {', '.join(REPLAYABLE)}; use a plain reset --stage {stage}")
    # Snapshot before any mutations so reparse is atomic: on any exception,
    # the record is restored to exactly its pre-call state before propagating.
    snapshot = copy.deepcopy(record)
    try:
        width, height = record["image_size"]
        mode = (record.get("provenance") or {}).get("coord_mode", "thousand")   # records from before the seam: 0-1000
        if stage == "propose":
            return _reparse_propose(record, width, height, mode)
        fn = {"describe": _reparse_describe, "ground": _reparse_ground, "verify": _reparse_verify}[stage]
        changed = 0
        for cand in record["candidates"]:
            if fn(cand, width, height, mode):
                _clear_after(cand, stage)
                changed += 1
        if changed:
            keep = STAGES[:STAGES.index(stage) + 1]
            record["stages_done"] = [s for s in record["stages_done"] if s in keep]
            record["selected"], record["status"] = None, None
        return changed
    except BaseException:
        record.clear()
        record.update(snapshot)
        raise
