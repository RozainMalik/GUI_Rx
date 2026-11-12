from PIL import Image

from benchev.adapters.fake import FakeAdapter
from benchev.config import load_protocol
from benchev.geometry import area_crop_px
from benchev.inference import predict_sample


def test_one_step_resizes_for_sspro():
    a = FakeAdapter(lambda img, ins: (0.25, 0.75))
    out = predict_sample(a, Image.new("RGB", (3840, 2160)), "x", load_protocol("sspro@v1"), "1step")
    assert a.calls[0]["size"] == (3200, 1800) and a.calls[0]["max_pixels"] == 14_777_616
    assert out["point"] == (0.25, 0.75) and out["input_size"] == (3200, 1800)


def test_two_step_maps_back_like_ground_two_step():
    # first pass -> centre; second pass -> crop centre; fixed 616 crop on 1920x1080, zoom 2
    a = FakeAdapter(lambda img, ins: (0.5, 0.5))
    out = predict_sample(a, Image.new("RGB", (1920, 1080)), "x", load_protocol("medspot@v1"), "2step")
    assert out["crop_box"] == (652, 232, 1268, 848)
    assert a.calls[1]["size"] == (1232, 1232)
    assert out["point"] == ((652 + 0.5 * 616) / 1920, (232 + 0.5 * 616) / 1080)


def test_two_step_area_crop_uses_adapter_multiple():
    a = FakeAdapter(lambda img, ins: (0.5, 0.5), crop_multiple=32)
    out = predict_sample(a, Image.new("RGB", (3840, 2160)), "x", load_protocol("sspro@v1"), "2step")
    side = out["crop_box"][2] - out["crop_box"][0]
    assert side % 32 == 0 and a.calls[0]["size"] == (3840, 2160) and a.calls[0]["max_pixels"] == 5_760_000
    assert a.calls[1]["size"] == (side * 3, side * 3)


def test_two_step_none_first_point():
    a = FakeAdapter(lambda img, ins: None)
    out = predict_sample(a, Image.new("RGB", (800, 600)), "x", load_protocol("medspot@v1"), "2step")
    assert out["point"] is None and out["crop_box"] is None and len(a.calls) == 1


def test_oswg_two_step_crops_native_image_like_upstream():
    """upstream osworld-g_two_stage.py: first pass on the 2720x1530 resize, crop from the NATIVE image."""
    a = FakeAdapter(lambda img, ins: (0.5, 0.5), crop_multiple=28)
    out = predict_sample(a, Image.new("RGB", (1920, 1080)), "x", load_protocol("oswg@v1"), "2step")
    side = area_crop_px(1920, 1080, 616 ** 2, 5_760_000, 28)
    assert a.calls[0]["size"] == (2720, 1530)
    x0, y0, x1, y1 = out["crop_box"]
    assert (x1 - x0, y1 - y0) == (side, side) and x1 <= 1920 and y1 <= 1080
    assert a.calls[1]["size"] == (int(side * 2.5), int(side * 2.5))
    assert out["point"] == ((x0 + 0.5 * side) / 1920, (y0 + 0.5 * side) / 1080)


def test_predict_sample_uses_model_max_pixels():
    a = FakeAdapter(lambda img, ins: (0.5, 0.5))
    predict_sample(a, Image.new("RGB", (800, 600)), "x", load_protocol("medspot@v1"), "1step", max_pixels=1_000_000)
    assert a.calls[-1]["max_pixels"] == 1_000_000
    predict_sample(a, Image.new("RGB", (800, 600)), "x", load_protocol("medspot@v1"), "2step", max_pixels=1_000_000)
    assert [c["max_pixels"] for c in a.calls[-2:]] == [1_000_000, 1_000_000]
    predict_sample(a, Image.new("RGB", (800, 600)), "x", load_protocol("medspot@v1"), "1step")
    assert a.calls[-1]["max_pixels"] == 5_860_400
