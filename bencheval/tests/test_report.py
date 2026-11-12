from benchev.registry import Row
from benchev.report import build


def R(**kw):
    b = dict(model_id="ours-2b-ab-clin", bench="ssv2", split="all", mode="1step", protocol_id="ssv2@v1",
             metrics={"acc": 91.4}, n=1272, provenance="imported", source="s", parity="unverified")
    b.update(kw)
    return Row(**b)


def test_headline_and_markers():
    md = build([R(), R(mode="2step", metrics={"acc": 90.8}, provenance="harness", parity="n/a")])
    assert "91.4ⁱ" in md and "90.8 " in md


def test_unknown_and_partial_excluded():
    md = build([R(protocol_id="unknown", metrics={"acc": 11.1}),
                R(status="partial", provenance="harness", parity="n/a", metrics={"acc": 22.2})])
    main, other = md.split("## Other protocols / partial")
    assert "11.1" not in main and "22.2" not in main
    assert "11.1" in other and "22.2" in other


def test_medspot_shows_strict_and_tol():
    md = build([R(bench="medspot", protocol_id="medspot@v1", split="rule=strict", metrics={"TCA (%)": 43.06}, n=603),
                R(bench="medspot", protocol_id="medspot@v1", split="rule=tol14px", metrics={"TCA (%)": 49.07}, n=603)])
    assert "43.06ⁱ / 49.07ⁱ" in md


def test_harness_beats_import_in_same_cell():
    md = build([R(), R(provenance="harness", parity="n/a", metrics={"acc": 91.5})])
    main = md.split("## Other protocols / partial")[0]
    assert "91.5" in main and "91.4" not in main


def test_per_bench_split_table():
    md = build([R(), R(split="group=web", metrics={"acc": 88.0})])
    assert "## ssv2" in md and "group=web" in md and "88.0ⁱ" in md


def test_split_table_without_splits_has_valid_header():
    md = build([R(bench="ssv1", protocol_id="ssv1@v1")])
    sec = md.split("## ssv1 (1-step ‖ 2-step)")[1].strip().splitlines()
    assert sec[0] == "| model | all |" and sec[1] == "|---|---|"


def _ms(model, tca, **kw):
    b = dict(model_id=model, bench="medspot", split="rule=strict", protocol_id="medspot@v1", provenance="harness",
             parity="n/a", metrics={"TCA (%)": tca, "SHR (%)": 70.0, "step1_accuracy (%)": 60.0, "prefix_SHR (%)": 50.0},
             n=603)
    b.update(kw)
    return R(**b)


def _cfg(mid, kind="published", **kw):
    from benchev.config import ModelCfg
    return ModelCfg(id=mid, kind=kind, adapter="hf_chat", ckpt=None, base=None, status="published", **kw)


def test_table_lists_strict_harness_rows_only():
    from benchev.report import medspot_table
    rows = [_ms("m-a", 31.0), _ms("m-b", 12.0, split="rule=tol14px"), _ms("m-c", 13.0, provenance="paper",
            protocol_id="medspot-legacy-tol7px", split="rule=legacy"), _ms("m-d", 14.0, status="partial")]
    md = medspot_table(rows, {m: _cfg(m) for m in ("m-a", "m-b", "m-c", "m-d")})
    assert "| m-a |" in md and "31.0" in md
    assert "12.0" not in md and "13.0" not in md and "14.0" not in md


def test_table_separates_ours_and_sorts_by_tca():
    from benchev.report import medspot_table
    md = medspot_table([_ms("pub-lo", 20.0), _ms("pub-hi", 40.0), _ms("ours", 45.0), _ms("ours", 50.0, mode="2step")],
                       {"pub-lo": _cfg("pub-lo"), "pub-hi": _cfg("pub-hi"), "ours": _cfg("ours", kind="ours")})
    ours, comp = md.split("## Comparison models")
    assert "| ours | " in ours and "2step" in ours and "pub-" not in ours
    assert comp.index("pub-hi") < comp.index("pub-lo")


def test_table_flags():
    from benchev.report import medspot_table
    cfgs = {"a": _cfg("a", flags=["no-published-reference"]),
            "b": _cfg("b", max_pixels=1003520, parity={"bench": "ssv2", "published": 92.8, "ours": 90.1, "delta": -2.7,
                                                        "status": "miss"}),
            "c": _cfg("c", parity={"bench": "ssv1", "published": 84.7, "ours": 84.9, "delta": 0.2, "status": "pass"})}
    md = medspot_table([_ms("a", 1.0), _ms("b", 2.0), _ms("c", 3.0)], cfgs)
    line = {m: next(l for l in md.splitlines() if l.startswith(f"| {m} |")) for m in "abc"}
    assert "no-published-reference" in line["a"]
    assert "parity miss ssv2 -2.7" in line["b"] and "max_pixels=1003520" in line["b"]
    assert "parity ✓ ssv1 +0.2" in line["c"]


def test_table_shows_parse_fail_rate():
    from benchev.report import medspot_table
    md = medspot_table([_ms("a", 1.0)], {"a": _cfg("a")}, no_point=lambda r: 12.5)
    assert "12.5" in next(l for l in md.splitlines() if l.startswith("| a |"))
