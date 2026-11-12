from PIL import Image

from benchev.adapters import get_adapter
from benchev.adapters.recipes import RECIPES, parse

IMG = Image.new("RGB", (2000, 1000))


def _adapter(recipe):
    a = get_adapter("hf_chat")
    a.recipe = RECIPES[recipe]
    return a


def test_registered():
    assert get_adapter("hf_chat").name == "hf_chat"


def test_predict_abs_input_maps_back(monkeypatch):
    a = _adapter("ui_tars15")
    monkeypatch.setattr(a, "input_size", lambda image, mp: (1000, 500))
    monkeypatch.setattr(a, "_generate", lambda image, msgs, mp: "Action: click(point='<point>500 250</point>')")
    pt, raw = a.predict(IMG, "ok", 5_000_000)
    assert pt == (0.5, 0.5) and raw.startswith("Action")


def test_predict_k1000_ignores_input_size(monkeypatch):
    a = _adapter("uground")
    monkeypatch.setattr(a, "input_size", lambda image, mp: (1000, 500))
    monkeypatch.setattr(a, "_generate", lambda image, msgs, mp: "(250, 750)")
    assert a.predict(IMG, "ok", 5_000_000)[0] == (0.25, 0.75)


def test_predict_unparsable_keeps_raw(monkeypatch):
    a = _adapter("uground")
    monkeypatch.setattr(a, "input_size", lambda image, mp: (1, 1))
    monkeypatch.setattr(a, "_generate", lambda image, msgs, mp: "sorry")
    assert a.predict(IMG, "ok", 1) == (None, "sorry")


def test_predict_passes_max_pixels(monkeypatch):
    a, seen = _adapter("uground"), []
    monkeypatch.setattr(a, "input_size", lambda image, mp: seen.append(mp) or (1, 1))
    monkeypatch.setattr(a, "_generate", lambda image, msgs, mp: seen.append(mp) or "")
    a.predict(IMG, "ok", 1_806_336)
    assert seen == [1_806_336, 1_806_336]


def test_messages_layouts():
    m = _adapter("uground").messages("click OK", (1, 1), (1, 1))
    assert [c["type"] for c in m[-1]["content"]] == ["image", "text"] and "click OK" in m[-1]["content"][1]["text"]
    m = _adapter("mai_ui").messages("click OK", (1, 1), (1, 1))
    assert m[0]["role"] == "system" and [c["type"] for c in m[1]["content"]] == ["text", "image"]
    assert m[1]["content"][0]["text"] == "click OK\n"
    m = _adapter("ui_tars15").messages("click OK", (1, 1), (1, 1))
    assert [x["role"] for x in m] == ["user", "user"] and m[1]["content"] == [{"type": "image"}]


def test_qwen25vl_system_states_the_input_size():
    m = _adapter("qwen25vl").messages("x", (1288, 728), (1920, 1080))
    assert "The screen's resolution is 1288x728." in m[0]["content"][0]["text"]


def test_ui_tars_point_with_special_tokens_stripped():
    # <point> may be a special token that decoding drops
    assert parse("ui_tars", "Action: click(point='512 384')") == (512.0, 384.0)


def test_processor_passes_max_pixels_to_qwen_family_processors(monkeypatch):
    # transformers 5.x Qwen2VLImageProcessor: no max_pixels attribute, size is a SizeDict (not a dict)
    import sys
    import types
    seen = {}

    class SizeDict:
        longest_edge, shortest_edge = 16777216, 65536

    qwen_ip = types.SimpleNamespace(merge_size=2, patch_size=16, size=SizeDict())
    tf = types.SimpleNamespace(
        AutoImageProcessor=types.SimpleNamespace(from_pretrained=lambda ckpt: qwen_ip),
        AutoProcessor=types.SimpleNamespace(from_pretrained=lambda ckpt, **kw: seen.update(kw) or object()))
    monkeypatch.setitem(sys.modules, "transformers", tf)
    a = _adapter("mai_ui")
    a.ckpt, a._processors = "/w", {}
    a.processor(2_116_800)
    assert seen == {"max_pixels": 2_116_800}


def test_processor_leaves_other_processors_alone(monkeypatch):
    import sys
    import types
    seen = {}
    tf = types.SimpleNamespace(
        AutoImageProcessor=types.SimpleNamespace(from_pretrained=lambda ckpt: types.SimpleNamespace(size={"height": 560})),
        AutoProcessor=types.SimpleNamespace(from_pretrained=lambda ckpt, **kw: seen.update(kw) or object()))
    monkeypatch.setitem(sys.modules, "transformers", tf)
    a = _adapter("legacy_coords")
    a.ckpt, a._processors = "/w", {}
    a.processor(5_000_000)
    assert seen == {}


def _stub_transformers(monkeypatch, ip, seen):
    import sys
    import types
    tf = types.SimpleNamespace(
        AutoImageProcessor=types.SimpleNamespace(from_pretrained=lambda ckpt: ip),
        AutoProcessor=types.SimpleNamespace(from_pretrained=lambda ckpt, **kw: seen.update(kw) or object()))
    monkeypatch.setitem(sys.modules, "transformers", tf)


def test_processor_fixes_the_mistral_tokenizer_regex(monkeypatch):
    # Ministral-3's tokenizer ships a wrong pre-tokenizer regex; transformers warns to load it with fix_mistral_regex
    import types
    seen = {}
    _stub_transformers(monkeypatch, types.SimpleNamespace(size={"longest_edge": 1540}), seen)
    a = _adapter("legacy_coords")
    a.ckpt, a._processors, a.model_type = "/w", {}, "mistral3"
    a.processor(5_000_000)
    assert seen == {"fix_mistral_regex": True}
