import json
from collections import Counter

import pytest
from PIL import Image

from annotator.calling import Call
from annotator.frames import Frame
from oai import batchapi as ba
from tests.conftest import make_frame, run

MODELS = {k: "gpt-5.6-luna" for k in ("propose", "describe", "ground", "verify", "refine_ground")}
EFFORTS = {"propose": "low", "describe": "low", "ground": "none", "verify": "none", "refine_ground": "none"}


def a_frame(tmp_path):
    (tmp_path / "Alpha").mkdir(parents=True, exist_ok=True)
    make_frame().save(tmp_path / "Alpha" / "Alpha_001.png")
    return Frame("Alpha", "Alpha_001.png", 1, tmp_path / "Alpha" / "Alpha_001.png")


def test_request_key_is_stable_across_renders_and_sensitive_to_every_input(frame):
    call = Call(system="S", user_text="U", images=(frame,), thinking=False, kind="ground")
    d = [ba.image_digest(frame)]
    key = ba.request_key(call, "gpt-5.6-luna", "none", d)
    assert len(key) == 64 and key == ba.request_key(Call(system="S", user_text="U", images=(make_frame(),), kind="ground"), "gpt-5.6-luna", "none", d)
    assert ba.image_digest(make_frame()) == d[0] and ba.image_digest(make_frame(100, 50)) != d[0]
    others = [ba.request_key(call, "gpt-5.6-terra", "none", d), ba.request_key(call, "gpt-5.6-luna", "low", d),
              ba.request_key(call.with_thinking(True), "gpt-5.6-luna", "none", d),
              ba.request_key(call.with_repair("raw", "err"), "gpt-5.6-luna", "none", d),
              ba.request_key(Call(system="S", user_text="U2", images=(frame,), kind="ground"), "gpt-5.6-luna", "none", d),
              ba.request_key(Call(system="S", user_text="U", images=(frame,), kind="verify"), "gpt-5.6-luna", "none", d),
              ba.request_key(call, "gpt-5.6-luna", "none", [ba.image_digest(make_frame(100, 50))])]
    assert len(set(others) | {key}) == len(others) + 1 and ba.custom_id_of(key) == key[:32]


def test_reply_cache_replays_counts_errors_per_job_and_resets(tmp_path):
    cache = ba.ReplyCache(tmp_path / "replies.jsonl")
    cache.add_reply("k1", '{"boxes": []}', False, {"input": 10, "output": 2, "model": "gpt-5.6-luna"})
    assert cache.add_error("k2", "invalid image", "batch_1") and not cache.add_error("k2", "invalid image", "batch_1")
    assert cache.add_error("k2", "invalid image", "batch_2")
    again = ba.ReplyCache(tmp_path / "replies.jsonl")
    assert again.replies == {"k1": ('{"boxes": []}', False)} and again.errors["k2"] == 2 and again.last_error["k2"] == "invalid image"
    again.reset_errors()
    assert again.errors["k2"] == 0 and ba.ReplyCache(tmp_path / "replies.jsonl").errors == Counter() and again.replies == {"k1": ('{"boxes": []}', False)}


def test_collector_buffers_per_frame_rolls_parts_and_dedupes(tmp_path):
    col = ba.Collector(tmp_path / "round1", max_lines=2)
    body = {"model": "gpt-5.6-luna", "input": []}
    for i in range(3):
        col.add(f"k{i}" * 32, body, app="Alpha", frame="Alpha_001.png", kind="ground", model="gpt-5.6-luna", effort="none")
    col.add("k0" * 32, body, app="Alpha", frame="Alpha_001.png", kind="ground", model="gpt-5.6-luna", effort="none")   # duplicate
    col.add("z1" * 32, body, app="Beta", frame="Beta_001.png", kind="propose", model="gpt-5.6-luna", effort="low")
    assert col.count == 0 and not col.parts                                     # nothing written until a frame stops
    col.flush("Alpha", "Alpha_001.png")
    col.drop("Beta", "Beta_001.png")
    col.close()
    assert col.count == 3 and col.kinds == Counter(ground=3) and [p.name for p in col.parts] == ["round1.part1.jsonl", "round1.part2.jsonl"]
    lines = [json.loads(l) for p in col.parts for l in p.read_text(encoding="utf-8").splitlines()]
    assert [l["custom_id"] for l in lines] == ["k0" * 16, "k1" * 16, "k2" * 16] and all(l["method"] == "POST" and l["url"] == "/v1/responses" for l in lines)
    assert col.index["k1" * 16] == {"key": "k1" * 32, "app": "Alpha", "frame": "Alpha_001.png", "kind": "ground", "model": "gpt-5.6-luna", "effort": "none"}
    assert "z1" * 32 not in col.keys
    col.discard()
    assert not any(p.exists() for p in col.parts)


def test_deferred_generator_replays_records_or_poisons(tmp_path):
    fr = a_frame(tmp_path)
    cache, col = ba.ReplyCache(tmp_path / "replies.jsonl"), ba.Collector(tmp_path / "round1")
    gen = ba.DeferredGenerator(fr, cache, col, MODELS, EFFORTS)
    full = Image.open(fr.path).convert("RGB")
    call = Call(system="S", user_text="U", images=(full, full.crop((0, 0, 100, 50))), thinking=False, kind="verify")
    with pytest.raises(ba.Deferred):
        run(gen.generate(call))
    col.flush(fr.app, fr.name)
    col.close()
    line = json.loads(col.parts[0].read_text(encoding="utf-8"))
    content = line["body"]["input"][0]["content"]
    assert content[0] == {"type": "input_image", "file_id": "@frame:Alpha/Alpha_001.png", "detail": "high"}     # the raw frame: by reference
    assert content[1]["image_url"].startswith("data:image/png;base64,") and line["body"]["model"] == "gpt-5.6-luna"
    key = col.index[line["custom_id"]]["key"]
    cache.add_reply(key, '{"verdict": "correct"}', False, {"input": 1, "output": 1, "model": "gpt-5.6-luna"})
    reply = run(gen.generate(call))
    assert reply.content == '{"verdict": "correct"}' and reply.usage is None
    cache2 = ba.ReplyCache(tmp_path / "r2.jsonl")
    cache2.add_error(key, "invalid image", "batch_1")
    gen2 = ba.DeferredGenerator(fr, cache2, ba.Collector(tmp_path / "round2"), MODELS, EFFORTS)
    with pytest.raises(ba.Deferred):                                             # one error: try again
        run(gen2.generate(call))
    cache2.add_error(key, "invalid image", "batch_2")
    with pytest.raises(ba.Poisoned, match="invalid image"):
        run(gen2.generate(call))


def test_poisoned_file_round_trips_and_clear_resets_the_cache_errors(tmp_path):
    cache = ba.ReplyCache(tmp_path / "replies.jsonl")
    cache.add_error("k", "boom", "batch_1")
    assert ba.load_poisoned(tmp_path) == {}
    ba.save_poisoned(tmp_path, {"Alpha/Alpha_001.png": "describe failed 2 rounds: boom"})
    assert ba.load_poisoned(tmp_path) == {"Alpha/Alpha_001.png": "describe failed 2 rounds: boom"}
    assert ba.clear_poisoned(tmp_path, cache) == 1 and ba.load_poisoned(tmp_path) == {} and cache.errors["k"] == 0


# --- the pass -------------------------------------------------------------------------------------

from annotator.records import load_record, record_path
from oai import batches as b
from oai.config import load_config
from tests.conftest import (DESCRIBE_MARK, GOOD_REPLIES, REGION_PX, TARGET_PX, happy_router, make_data, user_text_of,
                            write_config)


def setup(tmp_path, apps=("Alpha", "Beta"), n=1, size=2):
    cfg = load_config(write_config(tmp_path))
    make_data(cfg.data, apps=apps, n=n)
    plan = b.load_plan(cfg.work / "batches.json", cfg.seed)
    return cfg, b.plan_batch(plan, b.corpus_order(cfg.data, cfg.seed), size)


def answer(result, cache, router):
    """Feed every collected request through `router` into the cache - what a Batch job does."""
    for part in result.parts:
        for line in part.read_text(encoding="utf-8").splitlines():
            req = json.loads(line)
            entry = result.index[req["custom_id"]]
            reply = router(req["body"])
            if isinstance(reply, dict):
                reply = json.dumps(reply)
            cache.add_reply(entry["key"], reply, False, {"input": 2448, "output": 40, "model": entry["model"], "effort": entry["effort"]})


def rounds_of(tmp_path, cfg, batch, router, cache=None, max_rounds=10):
    cache = cache or ba.ReplyCache(tmp_path / "replies.jsonl")
    poisoned, kinds = {}, []
    for n in range(1, max_rounds + 1):
        res = run(ba.run_pass(cfg, batch, cache, poisoned, tmp_path / f"round{n}", log=lambda *_: None))
        if res.count == 0:
            break
        kinds.append(res.kinds)
        answer(res, cache, router)
    return kinds, poisoned


def test_a_pass_collects_every_sibling_call_of_a_stage(tmp_path):
    cfg, batch = setup(tmp_path)
    kinds, poisoned = rounds_of(tmp_path, cfg, batch, happy_router())
    assert kinds == [Counter(propose=2), Counter(describe=2), Counter(ground=6), Counter(verify=2)] and poisoned == {}
    for app, name, index in batch.frames:
        assert load_record(record_path(cfg.work, app, index))["status"] == "shipped"
    assert not (tmp_path / "round5.part1.jsonl").exists()


def two_candidate_router():
    """Propose names two rows; describe answers by the element it was asked about."""
    from annotator.geometry import full_space
    from annotator.imaging import make_crop, region_window
    full = full_space(1920, 1080, "pixels")
    crop_space = make_crop(make_frame(), region_window(REGION_PX, 1920, 1080, 0.05)).space("pixels")
    target2 = (288, 222, 576, 276)
    base = happy_router()

    def route(body):
        t = user_text_of(body)
        if "Step 1 - Intent" in t:
            c1 = {**GOOD_REPLIES["propose"]["candidates"][0], "bbox": full.from_px(TARGET_PX)}
            c2 = {**c1, "rank": 2, "element_name": "Thumbnails", "text_on_screen": "Thumbnails", "bbox": full.from_px(target2)}
            return json.dumps({**GOOD_REPLIES["propose"], "focus_region": {"bbox": full.from_px(REGION_PX), "kind": "menu", "why": "w"},
                               "candidates": [c1, c2]})
        if DESCRIBE_MARK in t and "element_name:   Thumbnails" in t:
            instr = {"functional": "Show the images as thumbnails.", "visual": "Click the menu item labelled 'Thumbnails'.",
                     "spatial": "Click the seventh row of the open File menu, just above 'Slideshow...'."}
            return json.dumps({**GOOD_REPLIES["describe"], "bbox": crop_space.from_px(target2), "text_on_screen": "Thumbnails", "instructions": instr})
        return base(body)
    return route


def test_nested_ground_fan_out_lands_in_one_round(tmp_path):
    cfg, batch = setup(tmp_path, apps=("Alpha",), n=1, size=1)
    kinds, _ = rounds_of(tmp_path, cfg, batch, two_candidate_router())
    assert kinds[:3] == [Counter(propose=1), Counter(describe=2), Counter(ground=6)]     # 2 candidates x 3 variants, one round


def test_a_poisoned_frame_is_set_aside_and_its_pending_requests_dropped(tmp_path):
    cfg, batch = setup(tmp_path)
    cache = ba.ReplyCache(tmp_path / "replies.jsonl")
    res = run(ba.run_pass(cfg, batch, cache, {}, tmp_path / "round1", log=lambda *_: None))
    answer(res, cache, happy_router())
    res = run(ba.run_pass(cfg, batch, cache, {}, tmp_path / "round2", log=lambda *_: None))
    router = happy_router()
    for part in res.parts:
        for line in part.read_text(encoding="utf-8").splitlines():
            req = json.loads(line)
            e = res.index[req["custom_id"]]
            if e["app"] == "Alpha":
                cache.add_error(e["key"], "invalid image", "batch_2")
                cache.add_error(e["key"], "invalid image", "batch_3")
            else:
                cache.add_reply(e["key"], router(req["body"]), False, {"input": 2448, "output": 40, "model": e["model"]})
    poisoned, logs = {}, []
    res = run(ba.run_pass(cfg, batch, cache, poisoned, tmp_path / "round3", log=logs.append))
    assert list(poisoned) == ["Alpha/Alpha_001.png"] and "invalid image" in poisoned["Alpha/Alpha_001.png"]
    assert res.poisoned_now == ["Alpha/Alpha_001.png"] and res.kinds == Counter(ground=3)     # Beta's ground only
    assert any(l.startswith("Alpha/Alpha_001.png: POISONED") for l in logs)
    res = run(ba.run_pass(cfg, batch, cache, poisoned, tmp_path / "round4", log=logs.append))
    assert Counter(e["app"] for e in res.index.values()) == Counter(Beta=3)                    # unanswered: collected again; Alpha skipped


def test_run_pass_rolls_parts_at_the_configured_size(tmp_path, monkeypatch):
    cfg, batch = setup(tmp_path)
    seen = {}
    real = ba.Collector

    def spy(prefix, **kw):
        seen.update(kw)
        return real(prefix, **kw)
    monkeypatch.setattr(ba, "Collector", spy)
    run(ba.run_pass(cfg, batch, ba.ReplyCache(tmp_path / "r.jsonl"), {}, tmp_path / "round1", log=lambda *_: None))
    assert seen == {"max_bytes": cfg.part_mb * 1_000_000}
