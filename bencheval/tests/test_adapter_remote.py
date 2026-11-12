import pytest
from PIL import Image

from benchev.adapters import get_adapter
from benchev.adapters.recipes import RECIPES

IMG = Image.new("RGB", (2000, 1000))


@pytest.mark.parametrize("name,recipe,answer,point", [
    ("deepseek_vl2", "deepseek_ref", "<|ref|>OK<|/ref|><|det|>[[0,0,999,499.5]]<|/det|>", (0.5, 0.25)),
    ("cogagent", "cogagent", "Grounded Operation: CLICK(box=[[200,400,300,600]], element_info='OK')", (0.25, 0.5)),
    ("seeclick", "seeclick", "(0.25,0.75)", (0.25, 0.75)),
])
def test_predict_maps_answer(monkeypatch, name, recipe, answer, point):
    a = get_adapter(name)
    assert a.name == name
    a.recipe, seen = RECIPES[recipe], {}
    monkeypatch.setattr(a, "_generate", lambda image, text: seen.update(text=text) or answer)
    pt, raw = a.predict(IMG, "click OK", 5_000_000)
    assert pt == pytest.approx(point) and raw == answer and "click OK" in seen["text"]


@pytest.mark.parametrize("name,recipe", [("deepseek_vl2", "deepseek_ref"), ("cogagent", "cogagent"), ("seeclick", "seeclick")])
def test_garbage_is_none(monkeypatch, name, recipe):
    a = get_adapter(name)
    a.recipe = RECIPES[recipe]
    monkeypatch.setattr(a, "_generate", lambda image, text: "no idea")
    assert a.predict(IMG, "x", 1) == (None, "no idea")


def test_deepseek_loads_in_bf16_not_fp32(monkeypatch):
    # transformers 4.38 loads 16B params in fp32 (~64 GB host RAM) unless torch_dtype is given
    import sys
    import types
    seen = {}

    class Model:
        def to(self, *a, **k):
            return self

        def eval(self):
            return self

    torch = types.SimpleNamespace(bfloat16="bf16")
    tf = types.SimpleNamespace(AutoModelForCausalLM=types.SimpleNamespace(
        from_pretrained=lambda ckpt, **kw: seen.update(kw) or Model()))
    dsm = types.SimpleNamespace(DeepseekVLV2Processor=types.SimpleNamespace(from_pretrained=lambda ckpt: object()))
    for name, mod in {"torch": torch, "transformers": tf, "deepseek_vl2": types.SimpleNamespace(models=dsm),
                      "deepseek_vl2.models": dsm}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    from benchev.config import ModelCfg
    get_adapter("deepseek_vl2").load(ModelCfg(id="d", kind="published", adapter="deepseek_vl2", ckpt="/w", base=None,
                                              status="published", recipe="deepseek_ref"), "cuda")
    assert seen.get("torch_dtype") == "bf16" and seen.get("low_cpu_mem_usage") is True
