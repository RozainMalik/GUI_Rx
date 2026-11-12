import pytest

from benchev.adapters import get_adapter


def test_registered():
    a = get_adapter("text_coords")
    assert a.name == "text_coords" and a.crop_multiple == 32


@pytest.mark.parametrize("point,expected", [((0.25, 0.5), (0.25, 0.5)), (None, None)])
def test_predict_returns_point_and_raw(monkeypatch, point, expected):
    tc = pytest.importorskip("gui_aima.qwen35.text_coord")
    seen = {}

    def fake_ground(model, proc, image, instruction, max_pixels, prompt="tool", **kw):
        seen.update(image=image, instruction=instruction, max_pixels=max_pixels, prompt=prompt)
        return {"point": point, "raw": "RAW", "point_if_pixels": None, "input_wh": (1, 1)}

    monkeypatch.setattr(tc, "ground_text", fake_ground)
    a = get_adapter("text_coords")
    a.model, a.processor = object(), object()
    assert a.predict("IMG", "click ok", 123) == (expected, "RAW")
    assert seen == {"image": "IMG", "instruction": "click ok", "max_pixels": 123, "prompt": "tool"}


def test_parse_point_tool_call_is_0_1000():
    tc = pytest.importorskip("gui_aima.qwen35.text_coord")
    assert tc.parse_point('<tool_call>{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [500, 250]}}') == (500.0, 250.0)
    assert tc.parse_point("no numbers here") is None
