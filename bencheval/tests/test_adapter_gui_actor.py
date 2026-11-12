import json
import sys
import types

import pytest

from benchev.adapters import get_adapter
from benchev.adapters import gui_actor as ga


def test_registered():
    a = get_adapter("gui_actor")
    assert a.name == "gui_actor" and a.crop_multiple == 28


@pytest.mark.parametrize("model_type", ["qwen2_vl", "qwen2_5_vl"])
def test_backbone_from_config(tmp_path, model_type):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    assert ga.backbone(str(tmp_path)) == model_type
    assert model_type in ga.SYSTEM and model_type in ga.CLASSES


def test_backbone_unknown_raises(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llava"}), encoding="utf-8")
    with pytest.raises(ValueError, match="llava"):
        ga.backbone(str(tmp_path))


def test_system_messages_match_upstream_eval():
    assert ga.SYSTEM["qwen2_vl"].startswith("You are a GUI agent. You are given a task and a screenshot")
    assert ga.SYSTEM["qwen2_5_vl"].startswith("You are a GUI agent. Given a screenshot of the current GUI")
    assert ga.CLASSES["qwen2_vl"] == ("gui_actor.modeling", "Qwen2VLForConditionalGenerationWithPointer")
    assert ga.CLASSES["qwen2_5_vl"] == ("gui_actor.modeling_qwen25vl", "Qwen2_5_VLForConditionalGenerationWithPointer")


def test_conversation_layout():
    c = ga.conversation("IMG", "click ok", "SYS")
    assert c == [{"role": "system", "content": [{"type": "text", "text": "SYS"}]},
                 {"role": "user", "content": [{"type": "image", "image": "IMG"}, {"type": "text", "text": "click ok"}]}]


def test_use_actor_package_idempotent():
    before = sys.path.count(ga.UPSTREAM)
    try:
        ga.use_actor_package()
        ga.use_actor_package()
        assert sys.path.count(ga.UPSTREAM) == 1 and sys.path[0] == ga.UPSTREAM
    finally:
        while sys.path.count(ga.UPSTREAM) > before:
            sys.path.remove(ga.UPSTREAM)


def _fake_inference_module(monkeypatch, result, seen):
    def inference(conv, model, tok, proc, logits_processor=None, use_placeholder=False, topk=5):
        seen.update(conv=conv, use_placeholder=use_placeholder, topk=topk, logits_processor=logits_processor)
        return result
    pkg = types.ModuleType("gui_actor")
    mod = types.ModuleType("gui_actor.inference")
    mod.inference = inference
    monkeypatch.setitem(sys.modules, "gui_actor", pkg)
    monkeypatch.setitem(sys.modules, "gui_actor.inference", mod)


def _adapter(backbone="qwen2_5_vl"):
    a = get_adapter("gui_actor")
    a.model, a.backbone = object(), backbone
    a._processors = {123: types.SimpleNamespace(tokenizer="TOK")}
    return a


def test_predict_returns_top1(monkeypatch):
    pytest.importorskip("torch")
    seen = {}
    _fake_inference_module(monkeypatch, {"topk_points": [(0.25, 0.75), (0.1, 0.1)], "output_text": "RAW"}, seen)
    assert _adapter("qwen2_vl").predict("IMG", "click ok", 123) == ((0.25, 0.75), "RAW")
    assert seen["use_placeholder"] is True and seen["topk"] == 3 and seen["logits_processor"] is None
    assert seen["conv"][0]["content"][0]["text"] == ga.SYSTEM["qwen2_vl"]


@pytest.mark.parametrize("pts", [[], None])
def test_predict_no_points_is_miss(monkeypatch, pts):
    pytest.importorskip("torch")
    _fake_inference_module(monkeypatch, {"topk_points": pts, "output_text": None}, {})
    assert _adapter().predict("IMG", "x", 123) == (None, "")


def test_processor_cached_per_max_pixels(monkeypatch):
    calls = []
    fake_tf = types.ModuleType("transformers")

    class AutoProcessor:
        @staticmethod
        def from_pretrained(path, max_pixels):
            calls.append(max_pixels)
            return ("P", max_pixels)

    fake_tf.AutoProcessor = AutoProcessor
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    a = get_adapter("gui_actor")
    a.ckpt, a._processors = "CKPT", {}
    assert a.processor(14777616) == ("P", 14777616)
    assert a.processor(5760000) == ("P", 5760000)
    assert a.processor(14777616) == ("P", 14777616)
    assert calls == [14777616, 5760000]
