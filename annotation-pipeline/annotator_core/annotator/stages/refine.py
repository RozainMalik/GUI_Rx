"""S5 - refine: RegionFocus crop-and-re-ground with two changes - crops are centred on the proposer's S2 box (never the rejected consensus), and the
prompt states what was wrong. Crop votes choose the box but never count toward agreement.
Then S4 runs once more on the final box; if the read-back label moved, S2 rewrites the
instructions anchored to it. At most one round, <= 8 calls (6 grounds + verify + optional S2)."""
from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import asdict, dataclass

from ..geometry import center, median_box
from ..imaging import FOCUS_FRACTIONS, focus_windows, make_crop, verify_window
from ..prompts import REFINE_LOOSE_SENTENCE, REFINE_UNKNOWN_SENTENCE, REFINE_WRONG_SENTENCE, render
from ..selection import pick_cluster
from ..validators import VARIANTS, normalise_label
from . import describe as s2
from . import verify as s4
from .ground import ground_crop
from .propose import CandidateSeed

CROP_VARIANTS = ("functional", "visual")
VERDICT_RANK = {"correct": 0, "loose": 1, "partial": 2, "wrong_element": 3}


@dataclass
class Refinement:
    focal_px: list
    windows: list
    crops: list                     # per crop call: window, fraction, variant, bbox_1000, bbox_px, box_sane, raw, empty, error, calls
    pool: dict                      # vote name -> bbox_px
    cluster: list
    final_bbox_px: list | None
    reverify: dict | None
    rewrite: dict | None            # the S2 rewrite's Description as a dict, when it ran and parsed
    instructions_changed: list
    calls: int
    outcome: str                    # fixed | unchanged | moved | worsened | no_box


def previous_sentence(verify) -> str:
    """What the re-grounding prompt says about the rejected attempt."""
    if not verify or verify.get("verdict") is None:
        return REFINE_UNKNOWN_SENTENCE
    if verify["verdict"] in ("loose", "partial"):
        return REFINE_LOOSE_SENTENCE
    if verify.get("text_in_box"):
        return render(REFINE_WRONG_SENTENCE, text_in_box=verify["text_in_box"])
    return REFINE_UNKNOWN_SENTENCE


def vote(pool: dict, verifier_key="verifier"):
    """(final box, cluster): per-coordinate median of the largest agreement cluster; ties ->
    the cluster holding the verifier's box, then the smaller median box."""
    cluster = pick_cluster(pool, prefer=verifier_key)
    return median_box(pool[k] for k in cluster), cluster


async def refine(gen, frame, app, region_kind, seed: CandidateSeed, describe: dict, ground: dict,
                 verify, consensus, mode="thousand") -> Refinement:
    width, height = frame.size
    focal = center(tuple(describe["bbox_px"]))
    windows = focus_windows(focal, width, height)
    sentence = previous_sentence(verify)
    jobs = []
    for fraction, win in zip(FOCUS_FRACTIONS, windows):
        crop = make_crop(frame, win)
        for variant in CROP_VARIANTS:
            jobs.append((fraction, win, variant, ground_crop(
                gen, crop, app, describe["instructions"][variant], describe["text_on_screen"],
                describe["icon"], sentence, (width, height), mode)))
    results = await asyncio.gather(*(j[3] for j in jobs))

    calls, crops, pool = 0, [], {}
    for (fraction, win, variant, _), g in zip(jobs, results):
        calls += g.calls
        crops.append({"window": list(win), "fraction": fraction, "variant": variant, "bbox_1000": g.bbox_1000,
                      "bbox_px": g.bbox_px, "box_sane": g.box_sane, "raw": g.raw, "empty": g.empty,
                      "error": g.error, "calls": g.calls})
        if g.bbox_px and not g.empty and g.error is None and g.box_sane is None:
            pool[f"crop_{fraction}_{variant}"] = tuple(g.bbox_px)
    if verify and verify.get("corrected_bbox_px") and verify.get("corrected_box_sane") is None:
        pool["verifier"] = tuple(verify["corrected_bbox_px"])
    if describe.get("bbox_px") and describe.get("box_sane") is None:
        pool["describe"] = tuple(describe["bbox_px"])
    for variant in VARIANTS:
        gv = (ground or {}).get(variant) or {}
        if gv.get("bbox_px") and gv.get("box_sane") is None:
            pool[variant] = tuple(gv["bbox_px"])

    base = Refinement(focal_px=[round(focal[0]), round(focal[1])], windows=[list(w) for w in windows],
                      crops=crops, pool={k: list(v) for k, v in pool.items()}, cluster=[], final_bbox_px=None,
                      reverify=None, rewrite=None, instructions_changed=[], calls=calls, outcome="no_box")
    if not pool:
        return base

    final, cluster = vote(pool)
    reverify = await s4.verify(gen, frame, final, describe["instructions"], describe["text_on_screen"], mode)
    calls += reverify.calls

    rewrite, changed = None, []
    if (reverify.verified and reverify.text_in_box
            and normalise_label(reverify.text_in_box) != normalise_label(describe["text_on_screen"])):
        anchored = dataclasses.replace(seed, bbox_px=final, text_on_screen=reverify.text_in_box)
        rw = await s2.describe(gen, frame, app, region_kind, verify_window(final, width, height), anchored,
                               anchor_text=reverify.text_in_box, mode=mode)
        calls += rw.calls
        if rw.present and rw.error is None and rw.instructions:
            rewrite = asdict(rw)
            changed = [v for v in VARIANTS if rw.instructions[v] != describe["instructions"][v]]

    before = tuple(consensus["bbox_px"]) if consensus else None
    if reverify.verified:
        outcome = "fixed"
    elif before is not None and final == before:
        outcome = "unchanged"
    elif verify and VERDICT_RANK.get(reverify.verdict, 4) > VERDICT_RANK.get(verify.get("verdict"), 4):
        outcome = "worsened"
    else:
        outcome = "moved"
    return dataclasses.replace(base, cluster=cluster, final_bbox_px=list(final), reverify=asdict(reverify),
                               rewrite=rewrite, instructions_changed=changed, calls=calls, outcome=outcome)
