from annotator import selection as sel

D = (288, 162, 576, 216)
OFF = (288, 300, 576, 340)         # a different row: centres fall outside each other


def cand(describe_box=D, ground=None, verified=True, verdict="correct", errors=(), state="highlighted",
         rank=1, refine=None, status="ok"):
    ground = {"functional": D, "visual": D, "spatial": D} if ground is None else ground
    c = {"rank": rank, "element": {"state": state}, "status": status,
         "describe": {"bbox_px": list(describe_box), "box_sane": None,
                      "instructions": {"functional": "f", "visual": "v", "spatial": "s"},
                      "instruction_errors": list(errors), "calls": 1},
         "ground": {k: ({"bbox_px": list(b), "box_sane": None, "calls": 1} if b else
                        {"bbox_px": None, "box_sane": None, "calls": 2, "error": "x"}) for k, b in ground.items()},
         "verify": {"verified": verified, "verdict": verdict, "readback_match": True, "calls": 1},
         "refine": refine, "final": None}
    cons = sel.consensus(sel.independent_boxes(c))
    c["consensus"] = {"bbox_px": list(cons[0]), "agreement": len(cons[1]) / 4, "cluster": cons[1]} if cons else None
    return c


def test_independent_boxes_skip_missing_and_insane():
    c = cand(ground={"functional": D, "visual": None, "spatial": D})
    c["ground"]["spatial"]["box_sane"] = "too_large"
    assert set(sel.independent_boxes(c)) == {"describe", "functional"}


def test_agreement_is_always_out_of_four():
    assert sel.agreement(D, {"describe": D, "functional": D, "visual": OFF}) == 0.5
    assert sel.agreement(D, {}) == 0.0


def test_consensus_picks_largest_cluster_and_median():
    ind = {"describe": (288, 162, 576, 216), "functional": (290, 160, 578, 218), "visual": (286, 164, 574, 214), "spatial": OFF}
    box, cluster = sel.consensus(ind)
    assert cluster == ["describe", "functional", "visual"] and box == (288, 162, 576, 216)
    assert sel.consensus({}) is None


def test_consensus_tie_prefers_cluster_with_describe_box():
    ind = {"describe": D, "functional": OFF, "visual": (289, 301, 577, 341), "spatial": (289, 163, 577, 217)}
    _, cluster = sel.consensus(ind)              # two clusters of two
    assert "describe" in cluster and "spatial" in cluster


def test_tier_table():
    assert sel.tier(True, 1.0, True) == "A" and sel.tier(True, 0.75, True) == "A"
    assert sel.tier(True, 0.5, True) == "B"
    assert sel.tier(True, 0.25, True) == "C"
    assert sel.tier(False, 1.0, True) == "C"
    assert sel.tier(True, 1.0, False) == "C"


def test_finalize_before_refine_uses_consensus_and_verify():
    f = sel.finalize(cand(), 1920, 1080)
    assert f["tier"] == "A" and f["agreement"] == 1.0 and f["bbox_px"] == list(D) and f["verified"]
    assert f["bbox_pct"] == [15.0, 15.0, 15.0, 5.0] and f["calls"] == 5
    assert f["instructions_changed"] == [] and f["verdict"] == "correct" and f["readback_match"] is True


def test_finalize_after_refine_recounts_agreement_against_the_moved_box():
    refine = {"final_bbox_px": list(OFF), "reverify": {"verified": True, "verdict": "correct", "readback_match": True},
              "rewrite": None, "instructions_changed": [], "calls": 7}
    f = sel.finalize(cand(verified=False, verdict="wrong_element", refine=refine), 1920, 1080)
    assert f["bbox_px"] == list(OFF) and f["verified"] and f["agreement"] == 0.0 and f["tier"] == "C" and f["calls"] == 12


def test_finalize_uses_rewrite_instructions():
    rw = {"instructions": {"functional": "F2", "visual": "V2", "spatial": "S2"}, "instruction_errors": []}
    refine = {"final_bbox_px": list(D), "reverify": {"verified": True, "verdict": "correct"}, "rewrite": rw,
              "instructions_changed": ["functional"], "calls": 8}
    f = sel.finalize(cand(refine=refine), 1920, 1080)
    assert f["instructions"]["functional"] == "F2" and f["instructions_changed"] == ["functional"] and f["tier"] == "A"


def test_finalize_none_without_a_box():
    c = cand()
    c["consensus"] = None
    assert sel.finalize(c, 1920, 1080) is None
    c2 = cand()
    c2["describe"] = None
    assert sel.finalize(c2, 1920, 1080) is None


def test_provisional_tier():
    assert sel.provisional_tier(cand()) == "A"
    assert sel.provisional_tier(cand(verified=False)) == "C"
    assert sel.provisional_tier(cand(ground={"functional": D, "visual": OFF, "spatial": OFF})) == "B"
    c = cand()
    c["verify"] = None
    assert sel.provisional_tier(c) == "C"


def test_choose_orders_by_tier_agreement_verdict_state_rank():
    a = cand(rank=1, state="plain", ground={"functional": D, "visual": OFF, "spatial": OFF})    # tier B
    b = cand(rank=2, state="plain")                                                           # tier A
    c = cand(rank=3, state="highlighted")                                                     # tier A, non-plain wins
    for x in (a, b, c):
        x["final"] = sel.finalize(x, 1920, 1080)
    assert sel.choose([a, b, c]) == 2
    assert sel.choose([a, b]) == 1
    b["status"] = "unreadable"
    assert sel.choose([a, b]) == 0
    assert sel.choose([]) is None


def test_frame_status():
    good = cand()
    good["final"] = sel.finalize(good, 1920, 1080)
    low = cand(verified=False)
    low["final"] = sel.finalize(low, 1920, 1080)
    assert sel.frame_status({"propose": {"rejected_reason": "no_candidates"}, "candidates": [], "selected": None}) == "abstained"
    assert sel.frame_status({"propose": {}, "candidates": [good], "selected": 0}) == "shipped"
    assert sel.frame_status({"propose": {}, "candidates": [low], "selected": 0}) == "lowconf"
    assert sel.frame_status({"propose": {}, "candidates": [{"status": "unreadable"}], "selected": None}) == "unreadable"
    assert sel.frame_status({"propose": {}, "candidates": [{"status": "dropped"}], "selected": None}) == "abstained"
