import asyncio
import json

import pytest

from annotator.calling import Call, Reply, TransportError
from annotator.prompts import prompt_hash
from annotator.records import STAGES, load_record, new_record, record_path, save_record
from annotator.runner import MixedPromptError, Routed, RunConfig, run_app
from tests.conftest import (GROUND_MARK, PROPOSE_MARK, VERIFY_MARK, FakeGenerator, happy_router, make_data, run)


def cfg_for(tmp_path, n=2, **kw):
    return RunConfig(work=tmp_path / "work", data_root=make_data(tmp_path, n), app="IrfanView", **kw)


def test_full_happy_path_ships_tier_a(tmp_path):
    cfg = cfg_for(tmp_path)
    logs = []
    summary = run(run_app(FakeGenerator(router=happy_router()), cfg, log=logs.append))
    assert summary["shipped"] == 2 and summary["frames"] == 2 and summary["calls"] == 12
    assert any(line.startswith("RECORDS_WRITTEN IrfanView frames=2") for line in logs)
    rec = load_record(record_path(cfg.work, "IrfanView", 1))
    assert rec["stages_done"] == list(STAGES) and rec["status"] == "shipped" and rec["selected"] == 0
    assert rec["frame"] == "IrfanView_001.png" and rec["image_size"] == [1920, 1080]
    c = rec["candidates"][0]
    assert c["status"] == "ok" and c["final"]["tier"] == "A" and c["final"]["agreement"] == 1.0
    assert c["refine"] is None and c["final"]["calls"] == 5 and c["final"]["bbox_px"] == [288, 162, 576, 216]
    assert c["consensus"]["cluster"] == ["describe", "functional", "visual", "spatial"]
    assert c["describe"]["raw"] and c["ground"]["visual"]["raw"] and c["verify"]["raw"] and rec["propose"]["raw"]
    assert rec["provenance"]["prompt_hash"] == prompt_hash() and rec["provenance"]["pipeline_version"] == "1.0.0"
    assert rec["provenance"]["thinking_fallbacks"] == 0


def test_rerun_makes_no_calls(tmp_path):
    cfg = cfg_for(tmp_path)
    run(run_app(FakeGenerator(router=happy_router()), cfg, log=lambda *_: None))

    def refuse(call):
        raise AssertionError("no calls expected on a finished record")
    summary = run(run_app(FakeGenerator(router=refuse), cfg, log=lambda *_: None))
    assert summary["shipped"] == 2 and summary["calls"] == 0


def test_transport_failure_leaves_stage_undone_and_a_rerun_resumes(tmp_path):
    cfg = cfg_for(tmp_path)
    base = happy_router()

    def flaky(call):
        if GROUND_MARK in call.user_text:
            raise TransportError("server gone")
        return base(call)
    summary = run(run_app(FakeGenerator(router=flaky), cfg, log=lambda *_: None))
    assert summary["transport_error"] == 2 and "shipped" not in summary
    rec = load_record(record_path(cfg.work, "IrfanView", 1))
    assert rec["stages_done"] == ["propose", "describe"] and rec["candidates"][0]["ground"] is None
    summary = run(run_app(FakeGenerator(router=base), cfg, log=lambda *_: None))
    assert summary["shipped"] == 2 and summary["calls"] == 8          # 3 groundings + 1 verify per frame


def test_judge_rejection_runs_refine_and_ships_lowconf(tmp_path):
    base = happy_router()

    def judge_says_no(call):
        if VERIFY_MARK in call.user_text:
            return json.dumps({"text_in_box": "Thumbnails", "element_in_box": "row", "verdict": "wrong_element",
                               "reason": "r", "corrected_bbox": None})
        return base(call)
    cfg = cfg_for(tmp_path, n=1)
    summary = run(run_app(FakeGenerator(router=judge_says_no), cfg, log=lambda *_: None))
    assert summary["lowconf"] == 1
    c = load_record(record_path(cfg.work, "IrfanView", 1))["candidates"][0]
    assert c["refine"]["outcome"] == "unchanged" and len(c["refine"]["crops"]) == 6
    assert c["final"]["tier"] == "C" and c["final"]["calls"] == 12 and c["final"]["verdict"] == "wrong_element"


def test_abstained_frame_is_final(tmp_path):
    base = happy_router()

    def no_candidates(call):
        if PROPOSE_MARK in call.user_text:
            return json.dumps({"intent": "i", "focus_region": {"bbox": [100, 100, 500, 400], "kind": "menu", "why": "w"}, "candidates": []})
        return base(call)
    cfg = cfg_for(tmp_path, n=1)
    assert run(run_app(FakeGenerator(router=no_candidates), cfg, log=lambda *_: None))["abstained"] == 1
    rec = load_record(record_path(cfg.work, "IrfanView", 1))
    assert rec["status"] == "abstained" and rec["stages_done"] == list(STAGES)
    assert rec["propose"]["rejected_reason"] == "no_candidates" and rec["selected"] is None


def test_dropped_and_unreadable_candidates(tmp_path):
    base = happy_router(two=True)

    def second_absent(call):
        if "ZOOMED CROP" in call.user_text and "Thumbnails" in call.user_text.split("\n", 8)[7]:
            return '{"element_present": false}'
        return base(call)
    cfg = cfg_for(tmp_path, n=1)
    assert run(run_app(FakeGenerator(router=second_absent), cfg, log=lambda *_: None))["shipped"] == 1
    rec = load_record(record_path(cfg.work, "IrfanView", 1))
    assert rec["selected"] == 0
    second = rec["candidates"][1]
    assert second["status"] == "dropped" and second["status_stage"] == "describe" and second["status_reason"] == "not_in_region"
    assert second["ground"] is None and second["final"] is None


def test_two_candidates_choose_best_and_early_stop_skips_refine(tmp_path):
    cfg = cfg_for(tmp_path, n=1)
    run(run_app(FakeGenerator(router=happy_router(two=True)), cfg, log=lambda *_: None))
    rec = load_record(record_path(cfg.work, "IrfanView", 1))
    assert rec["selected"] == 0 and rec["status"] == "shipped"
    assert rec["candidates"][0]["final"]["tier"] == "A" and rec["candidates"][1]["final"]["tier"] == "C"
    assert rec["candidates"][1]["refine"] is not None                       # refined because it was below A

    cfg2 = RunConfig(work=tmp_path / "work2", data_root=cfg.data_root, app="IrfanView", early_stop=True)
    run(run_app(FakeGenerator(router=happy_router(two=True)), cfg2, log=lambda *_: None))
    rec2 = load_record(record_path(cfg2.work, "IrfanView", 1))
    assert rec2["candidates"][1]["refine"] is None and rec2["selected"] == 0


def test_mixed_prompt_hash_refused_unless_allowed(tmp_path):
    cfg = cfg_for(tmp_path, n=1)
    stale = new_record("IrfanView", "IrfanView_001.png", 1, (1920, 1080), {"prompt_hash": "old", "thinking_fallbacks": 0})
    save_record(record_path(cfg.work, "IrfanView", 1), stale)
    with pytest.raises(MixedPromptError):
        run(run_app(FakeGenerator(router=happy_router()), cfg, log=lambda *_: None))
    cfg.allow_mixed = True
    assert run(run_app(FakeGenerator(router=happy_router()), cfg, log=lambda *_: None))["shipped"] == 1


def test_limit_and_indices(tmp_path):
    cfg = cfg_for(tmp_path, n=4, limit=2)
    assert run(run_app(FakeGenerator(router=happy_router()), cfg, log=lambda *_: None))["frames"] == 2
    assert record_path(cfg.work, "IrfanView", 1).exists() and record_path(cfg.work, "IrfanView", 3).exists()
    cfg2 = RunConfig(work=tmp_path / "w2", data_root=cfg.data_root, app="IrfanView", indices=[4])
    assert run(run_app(FakeGenerator(router=happy_router()), cfg2, log=lambda *_: None))["frames"] == 1
    assert record_path(cfg2.work, "IrfanView", 4).exists()


def test_server_death_trips_the_breaker(tmp_path):
    def dead(call):
        raise TransportError("server gone")
    cfg = cfg_for(tmp_path, n=12, frame_concurrency=1)
    gen = FakeGenerator(router=dead)
    summary = run(run_app(gen, cfg, log=lambda *_: None))
    assert summary["transport_error"] == 5 and summary["skipped"] == 7
    assert len(gen.calls) == 5                     # no calls dispatched for the skipped frames


def test_unexpected_exception_is_counted_not_fatal(tmp_path):
    def boom(call):
        raise RuntimeError("bug")
    cfg = cfg_for(tmp_path, n=1)
    logs = []
    summary = run(run_app(FakeGenerator(router=boom), cfg, log=logs.append))
    assert summary["error"] == 1 and any("RuntimeError" in line for line in logs)


def test_throttled_caps_in_flight_requests(tmp_path):
    base = happy_router()
    state = {"in_flight": 0, "peak": 0}

    async def counted(call):
        state["in_flight"] += 1
        state["peak"] = max(state["peak"], state["in_flight"])
        for _ in range(3):
            await asyncio.sleep(0)
        state["in_flight"] -= 1
        reply = base(call)
        return Reply(content=reply) if isinstance(reply, str) else reply

    class AsyncRouter:
        async def generate(self, call):
            return await counted(call)

    cfg = cfg_for(tmp_path, n=6, concurrency=3, frame_concurrency=8)
    summary = run(run_app(AsyncRouter(), cfg, log=lambda *_: None))
    assert summary["frames"] == 6 and summary["shipped"] == 6
    assert state["peak"] == 3


def test_routed_dispatches_by_call_kind():
    routed = Routed(FakeGenerator(["default"]), {"ground": FakeGenerator(["g"])})
    assert run(routed.generate(Call(system="", user_text="", kind="ground"))).content == "g"
    assert run(routed.generate(Call(system="", user_text="", kind="propose"))).content == "default"


def test_pixels_mode_end_to_end_and_provider_provenance(tmp_path):
    cfg = cfg_for(tmp_path, coord_mode="pixels", models={"propose": "m1"}, provenance_extra={"provider": "test", "batch": 7})
    gen = FakeGenerator(router=happy_router(mode="pixels"))
    summary = run(run_app(gen, cfg, log=lambda *_: None))
    assert summary["shipped"] == 2 and summary["calls"] == 12
    rec = load_record(record_path(cfg.work, "IrfanView", 1))
    prov = rec["provenance"]
    assert prov["coord_mode"] == "pixels" and prov["models"] == {"propose": "m1"} and prov["provider"] == "test" and prov["batch"] == 7
    assert prov["prompt_hash"] == prompt_hash("pixels") != prompt_hash()
    final = rec["candidates"][0]["final"]
    assert final["tier"] == "A" and all(abs(a - b) <= 1 for a, b in zip(final["bbox_px"], (288, 162, 576, 216)))
    assert {c.kind for c in gen.calls} == {"propose", "describe", "ground", "verify"}
    assert all("0-1000" not in c.system + c.user_text for c in gen.calls)


def test_thousand_records_keep_their_provenance_shape(tmp_path):
    cfg = cfg_for(tmp_path)
    run(run_app(FakeGenerator(router=happy_router()), cfg, log=lambda *_: None))
    prov = load_record(record_path(cfg.work, "IrfanView", 1))["provenance"]
    assert prov["coord_mode"] == "thousand" and "models" not in prov and "provider" not in prov


def test_generators_table_routes_one_stage_to_another_generator(tmp_path):
    cfg = cfg_for(tmp_path, n=1, generators={"ground": FakeGenerator(router=happy_router())})
    default = FakeGenerator(router=happy_router())
    run(run_app(default, cfg, log=lambda *_: None))
    assert {c.kind for c in default.calls} == {"propose", "describe", "verify"}
    assert {c.kind for c in cfg.generators["ground"].calls} == {"ground"}
