"""1-step / 2-step drivers. All protocol knobs come from the Protocol; the 2-step math equals
gui_aima.qwen35.inference.ground_two_step (crop_window + PIL default resize + linear map back)."""
from __future__ import annotations

from benchev.geometry import area_crop_px, crop_window, resize_to_area


def _crop_px(protocol, adapter, W, H):
    c = protocol.crop
    if c["kind"] == "fixed":
        return c["px"]
    return area_crop_px(W, H, c["portion"], c["ref_pixels"], adapter.crop_multiple)


def predict_sample(adapter, image, instruction, protocol, mode, max_pixels=None):
    """max_pixels: a model's own image limit, overriding the protocol's."""
    if mode == "zf":
        return _zoom_fusion(adapter, image, instruction, protocol, max_pixels)
    mp = max_pixels or protocol.max_pixels[mode]
    W, H = image.size
    in_size = resize_to_area(W, H, protocol.resize_to_pixels[mode])
    x_in = image if in_size == (W, H) else image.resize(in_size)
    p1, raw = adapter.predict(x_in, instruction, mp)
    out = {"point": p1, "raw": raw, "input_size": in_size, "first_point": p1, "crop_box": None}
    if mode == "1step" or p1 is None:
        return out
    # The crop is cut from the NATIVE image (first_point is normalised, so it carries over): upstream
    # osworld-g_two_stage.py keeps ori_image before its pre-resize and crops that; for SS-v2 / SS-Pro 2-step /
    # MedSPOT there is no 2-step pre-resize, so this equals ground_two_step on its `image` argument.
    w, h = W, H
    cp = _crop_px(protocol, adapter, W, H)
    x0, y0, x1, y1 = crop_window(w, h, p1, cp)
    crop = image.crop((x0, y0, x1, y1))
    crop = crop.resize((int(crop.width * protocol.zoom), int(crop.height * protocol.zoom)))
    p2, raw2 = adapter.predict(crop, instruction, mp)
    out.update(crop_box=(x0, y0, x1, y1), raw=raw2,
               point=None if p2 is None else ((x0 + p2[0] * (x1 - x0)) / w, (y0 + p2[1] * (y1 - y0)) / h))
    return out


def _zoom_fusion(adapter, image, instruction, protocol, max_pixels=None):
    """Zoom + full-view fusion (benchev.fusion). Pass 1 is exactly the 1-step pass (1-step pre-resize
    and max_pixels), so first_point equals the bench's 1-step click. Pass 2 is the 2-step zoom: the protocol's crop of
    the NATIVE image around that point, zoomed, at the 2-step max_pixels. The two grounding maps are fused for every λ
    (λ = 0: zoom map alone); `point` is λ = 1, the cross-fitted choice is made when the run is scored."""
    from benchev.fusion import LAMBDAS, fused_point
    W, H = image.size
    in_size = resize_to_area(W, H, protocol.resize_to_pixels["1step"])
    x_in = image if in_size == (W, H) else image.resize(in_size)
    p1, m1 = adapter.predict_map(x_in, instruction, max_pixels or protocol.max_pixels["1step"])
    out = {"point": p1, "raw": "", "input_size": in_size, "first_point": p1, "crop_box": None, "zoom_point": None,
           "fuse": None}
    if p1 is None or m1 is None:
        return out
    cp = _crop_px(protocol, adapter, W, H)
    x0, y0, x1, y1 = crop_window(W, H, p1, cp)
    crop = image.crop((x0, y0, x1, y1))
    crop = crop.resize((int(crop.width * protocol.zoom), int(crop.height * protocol.zoom)))
    p2, m2 = adapter.predict_map(crop, instruction, max_pixels or protocol.max_pixels["2step"])
    box = (x0, y0, x1, y1)
    out["crop_box"] = box
    if p2 is None or m2 is None:
        return out
    out["zoom_point"] = ((x0 + p2[0] * (x1 - x0)) / W, (y0 + p2[1] * (y1 - y0)) / H)
    out["fuse"] = {str(lam): list(fused_point(m1, m2, box, (W, H), lam)) for lam in (0,) + LAMBDAS}
    out["point"] = tuple(out["fuse"]["1.0"])
    return out
