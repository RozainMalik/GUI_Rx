"""Zero-shot probe: does base Qwen3.5 already ground GUI targets in its full-attention maps?"""
import torch
from qwen_vl_utils import process_vision_info

PROBE_SYSTEM = ("You are a GUI agent. Given a screenshot of the current GUI and a human instruction, "
                "locate the screen element that corresponds to the instruction and click on it.")
ANCHOR_PREFIX = "pyautogui.click("


def build_probe_inputs(processor, image, instruction, max_pixels):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": PROBE_SYSTEM}]},
        {"role": "user", "content": [{"type": "image", "image": image, "max_pixels": max_pixels},
                                     {"type": "text", "text": instruction}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) + ANCHOR_PREFIX
    images, _ = process_vision_info(messages, image_patch_size=processor.image_processor.patch_size)
    inputs = processor(text=[text], images=images, return_tensors="pt", return_mm_token_type_ids=True)
    ids = inputs["input_ids"][0]
    visual = torch.nonzero(ids == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")).squeeze(-1)
    anchor = len(ids) - 1
    merge = processor.image_processor.merge_size
    _, gh, gw = (inputs["image_grid_thw"][0] // torch.tensor([1, merge, merge])).tolist()
    return dict(inputs=inputs, anchor_index=anchor, query_indices=torch.arange(visual[-1] + 1, anchor),
                visual_indices=visual, grid_hw=(gh, gw), resized_wh=images[0].size)


def patch_center(idx, grid_w, patch_px=32):
    return (float((idx % grid_w) * patch_px + patch_px / 2), float((idx // grid_w) * patch_px + patch_px / 2))


def in_box(pt, box_xyxy):
    x, y = pt
    return box_xyxy[0] <= x <= box_xyxy[2] and box_xyxy[1] <= y <= box_xyxy[3]


def per_head_predictions(vis_attn):
    """Argmax visual patch of the anchor row (last row) for every (layer, head)."""
    return vis_attn[:, :, -1].argmax(-1)


def variant_predictions(vis_attn, sink_row, gates):
    """Predicted patch per aggregation variant. vis_attn: (L, H, R, V), anchor = last row; gates: (L, H)."""
    L, H, R, V = vis_attn.shape
    anchor = vis_attn[:, :, -1].reshape(L * H, V)
    sink_w = vis_attn[:, :, sink_row].sum(-1).reshape(-1).softmax(-1)            # GUI-AIMA Eq. 3
    gate_w = gates.reshape(-1) / gates.sum()
    return {
        "anchor_uniform": int(anchor.mean(0).argmax()),
        "anchor_sink_weighted": int((sink_w[:, None] * anchor).sum(0).argmax()),
        "anchor_gate_weighted": int((gate_w[:, None] * anchor).sum(0).argmax()),
        "allquery_uniform": int(vis_attn.mean((0, 1, 2)).argmax()),
    }
