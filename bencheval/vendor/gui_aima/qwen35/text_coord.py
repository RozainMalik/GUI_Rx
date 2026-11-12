"""Native text-coordinate grounding with the unmodified base Qwen3.5 (the no-GUI-AIMA baseline).

The model writes the click point as text. Qwen3-VL / Qwen3.5 agent coordinates are relative, 0-1000 on both axes
("The screen's resolution is 1000x1000"). Two prompts:
  tool   the chat template's own tool calling with a computer_use tool (Qwen's GUI-agent format)
  point  a plain request for {"point_2d": [x, y]} in 0-1000
Greedy decoding, thinking off, up to 256 new tokens (the 2B often writes a sentence before its tool call). parse_point() also records the raw text so a wrong convention shows up in the preds.
"""
import re

import torch
from qwen_vl_utils import process_vision_info

COMPUTER_USE_TOOL = {
    "type": "function",
    "function": {
        "name": "computer_use",
        "description": ("Use a mouse to interact with a computer.\n"
                        "* The screen's resolution is 1000x1000.\n"
                        "* Click with the cursor tip in the center of the element you want to click."),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["left_click"], "description": "The action to perform."},
                "coordinate": {"type": "array", "description": "(x, y): the pixel to click, each in 0-1000."},
            },
            "required": ["action", "coordinate"],
        },
    },
}
POINT_PROMPT = ("Locate the UI element described by the instruction and output only its click point as JSON: "
                '{{"point_2d": [x, y]}}, with x and y relative coordinates from 0 to 1000.\nInstruction: {}')

_NUM = r"(-?\d+(?:\.\d+)?)"


def parse_point(text):
    """(x, y) in the model's own units from the generated text, or None. Order: a 'coordinate'/'point_2d' pair,
    a 'bbox_2d' box (center), any [x1, y1, x2, y2] box (center), any (x, y) / [x, y] pair."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(rf"(?:coordinate|point_2d)\W*?{_NUM}\s*,\s*{_NUM}", text)
    if m:
        return float(m[1]), float(m[2])
    m = re.search(rf"[\[(]\s*{_NUM}\s*,\s*{_NUM}\s*,\s*{_NUM}\s*,\s*{_NUM}\s*[\])]", text)
    if m:
        x1, y1, x2, y2 = map(float, m.groups())
        return (x1 + x2) / 2, (y1 + y2) / 2
    m = re.search(rf"[\[(]\s*{_NUM}\s*,\s*{_NUM}\s*[\])]", text)
    return (float(m[1]), float(m[2])) if m else None


@torch.no_grad()
def ground_text(model, processor, image, instruction, max_pixels, prompt="tool", max_new_tokens=256,
                image_patch_size=16):
    """Return {"point": (x, y) in [0, 1] or None, "raw": generated text, "input_wh": model input size in px}.
    "point" reads the numbers as 0-1000; "point_if_pixels" reads them as pixels of the resized model input."""
    img = {"type": "image", "image": image, "max_pixels": max_pixels}
    if prompt == "tool":
        conv = [{"role": "user", "content": [img, {"type": "text", "text": instruction}]}]
        kw = {"tools": [COMPUTER_USE_TOOL]}
    else:
        conv = [{"role": "user", "content": [img, {"type": "text", "text": POINT_PROMPT.format(instruction)}]}]
        kw = {}
    text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                                         **kw)
    images, _ = process_vision_info(conv, image_patch_size=image_patch_size)
    x = processor(text=[text], images=images, return_tensors="pt").to(model.device)
    out = model.generate(**x, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                         top_k=None)
    raw = processor.batch_decode(out[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    _, gh, gw = x["image_grid_thw"][0].tolist()
    in_w, in_h = gw * image_patch_size, gh * image_patch_size
    p = parse_point(raw)
    clip = lambda v: min(max(v, 0.0), 1.0)
    return {"point": (clip(p[0] / 1000), clip(p[1] / 1000)) if p else None,
            "point_if_pixels": (clip(p[0] / in_w), clip(p[1] / in_h)) if p else None,
            "raw": raw, "input_wh": (in_w, in_h)}


def load_base_model(model_path, device="cpu"):
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    proc = AutoProcessor.from_pretrained(model_path)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, dtype=dtype)
    return model.to(device).eval(), proc
