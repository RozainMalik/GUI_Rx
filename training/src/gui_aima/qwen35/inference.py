"""Single-forward GUI-AIMA grounding on Qwen3.5 (placeholder pointer tokens; no generation)."""
import torch
from qwen_vl_utils import process_vision_info

from gui_aima.constants import (DEFAULT_POINTER_END_TOKEN, DEFAULT_POINTER_PAD_TOKEN, DEFAULT_POINTER_START_TOKEN,
                                chat_template, grounding_system_message)
from gui_aima.points import get_prediction_region_point
from gui_aima.qwen35.modeling import Qwen3_5ForConditionalGenerationWithPointer

ASSISTANT_STARTER = (f"<|im_start|>assistant<|recipient|>os\npyautogui.click("
                     f"{DEFAULT_POINTER_START_TOKEN}{DEFAULT_POINTER_PAD_TOKEN}{DEFAULT_POINTER_END_TOKEN})")


def load_pointer_model(ckpt, base=None, device="cpu", attention_args="query_1"):
    """Load a trained pointer checkpoint (or a LoRA adapter on `base`, merged) and its processor."""
    from transformers import AutoProcessor
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    proc = AutoProcessor.from_pretrained(ckpt)
    tok = proc.tokenizer
    if base:
        from peft import PeftModel
        model = Qwen3_5ForConditionalGenerationWithPointer.from_pretrained(base, dtype=dtype)
        model.resize_token_embeddings(len(tok))
        model = PeftModel.from_pretrained(model, ckpt).merge_and_unload()
    else:
        model = Qwen3_5ForConditionalGenerationWithPointer.from_pretrained(ckpt, dtype=dtype)
    model.config.pointer_start_token_id = tok.convert_tokens_to_ids(DEFAULT_POINTER_START_TOKEN)
    model.config.pointer_pad_token_id = tok.convert_tokens_to_ids(DEFAULT_POINTER_PAD_TOKEN)
    model.config.pointer_end_token_id = tok.convert_tokens_to_ids(DEFAULT_POINTER_END_TOKEN)
    model.set_attention_args(attention_args)
    return model.to(device).eval(), proc


def sspro_crop_px(width, height, portion=616 ** 2, ref_pixels=5760000, multiple=32):
    """Upstream ScreenSpot-Pro zoom crop: a square covering portion/ref_pixels of the image area,
    rounded down to whole visual tokens (upstream used 28 px; Qwen3.5 tokens are 32 px)."""
    return max(multiple, int((width * height * portion / ref_pixels) ** 0.5 / multiple) * multiple)


@torch.no_grad()
def ground(model, processor, image, instruction, max_pixels, image_patch_size=16):
    """Return the click point (x, y) in [0, 1] of `image`, the patch scores and the (h, w) patch grid."""
    conv = [
        {"role": "system", "content": [{"type": "text", "text": grounding_system_message}]},
        {"role": "user", "content": [{"type": "image", "image": image, "max_pixels": max_pixels},
                                     {"type": "text", "text": instruction}]},
    ]
    text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False,
                                         chat_template=chat_template) + ASSISTANT_STARTER
    images, _ = process_vision_info(conv, image_patch_size=image_patch_size)
    x = processor(text=[text], images=images, return_tensors="pt").to(model.device)
    out = model(**x, compute_pointer=True)
    scores = out.pointer_scores[0]
    merge = processor.image_processor.merge_size
    _, gh, gw = (x["image_grid_thw"][0].cpu() // torch.tensor([1, merge, merge])).tolist()
    point = get_prediction_region_point(scores, gw, gh, return_all_regions=False)
    return {"point": tuple(float(v) for v in point), "scores": scores, "grid_hw": (gh, gw)}


def ground_two_step(model, processor, image, instruction, max_pixels, crop_px=616, zoom=2.0, image_patch_size=16):
    """Paper Sec. 3.3: crop around the first prediction, zoom, re-ground, map back to the full image."""
    first = ground(model, processor, image, instruction, max_pixels, image_patch_size)
    W, H = image.size
    cx, cy = first["point"][0] * W, first["point"][1] * H
    half = crop_px / 2
    x0 = int(min(max(cx - half, 0), max(W - crop_px, 0)))
    y0 = int(min(max(cy - half, 0), max(H - crop_px, 0)))
    x1, y1 = min(x0 + crop_px, W), min(y0 + crop_px, H)
    crop = image.crop((x0, y0, x1, y1))
    crop = crop.resize((int(crop.width * zoom), int(crop.height * zoom)))
    second = ground(model, processor, crop, instruction, max_pixels, image_patch_size)
    px = (x0 + second["point"][0] * (x1 - x0)) / W
    py = (y0 + second["point"][1] * (y1 - y0)) / H
    return {"point": (px, py), "first": first, "second": second}
