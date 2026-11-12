"""Published GUI-Actor models via the official inference code (vendor/upstream_gui_actor, microsoft/GUI-Actor@d98d1bb).

Each call mirrors one sample of upstream eval/screenSpot_v2.py / screenSpot_pro.py: the backbone's system message,
`inference(..., use_placeholder=True, topk=3)` and the top-1 region point. Upstream passes its ForceFollowTokens
logits processor; with logits_processor=None `inference` builds the same one. The backbone (2B: Qwen2-VL, 3B/7B:
Qwen2.5-VL) comes from the checkpoint's config.json. The processor gets the protocol's max_pixels (upstream uses the
checkpoint default, 5,720,064; equal inputs on SS-v2, whose largest image is 5.18 MP).
Needs the upstream venv (torch 2.6, transformers 4.51.3, flash-attn 2.7.3, qwen-vl-utils 0.0.8), not the Qwen3.5 one.
"""
import json
import sys
from pathlib import Path

from benchev import ROOT
from benchev.adapters.base import Adapter
from benchev.adapters.gui_aima_upstream import last_position_head

UPSTREAM = str(ROOT / "vendor" / "upstream_gui_actor")
SYSTEM = {
    "qwen2_vl": ("You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series "
                 "of pyautogui actions to complete the task."),
    "qwen2_5_vl": ("You are a GUI agent. Given a screenshot of the current GUI and a human instruction, your task is to "
                   "locate the screen element that corresponds to the instruction. You should output a PyAutoGUI action "
                   "that performs a click on the correct position. To indicate the click location, we will use some "
                   "special tokens, which is used to refer to a visual patch later. For example, you can output: "
                   "pyautogui.click(<your_special_token_here>)."),
}
CLASSES = {"qwen2_vl": ("gui_actor.modeling", "Qwen2VLForConditionalGenerationWithPointer"),
           "qwen2_5_vl": ("gui_actor.modeling_qwen25vl", "Qwen2_5_VLForConditionalGenerationWithPointer")}


def backbone(ckpt):
    mt = json.loads((Path(ckpt) / "config.json").read_text(encoding="utf-8"))["model_type"]
    if mt not in CLASSES:
        raise ValueError(f"gui_actor: unsupported model_type {mt!r} in {ckpt}")
    return mt


def use_actor_package():
    if UPSTREAM in sys.path:
        sys.path.remove(UPSTREAM)
    sys.path.insert(0, UPSTREAM)


def conversation(image, instruction, system):
    return [{"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}]}]


class GuiActor(Adapter):
    name = "gui_actor"
    crop_multiple = 28      # patch 14, 2x2 merge

    def load(self, model_cfg, device):
        use_actor_package()
        import importlib

        import torch
        self.ckpt = model_cfg.ckpt
        self.backbone = backbone(self.ckpt)
        mod, cls = CLASSES[self.backbone]
        model_cls = getattr(importlib.import_module(mod), cls)
        self.model = model_cls.from_pretrained(self.ckpt, torch_dtype=torch.bfloat16, device_map=device,
                                               attn_implementation="flash_attention_2").eval()
        # Same memory measures as gui_aima_upstream: generation reads only the last position's logits and the pointer
        # comes from hidden states, so predictions are unchanged.
        self.model.lm_head = last_position_head(self.model.lm_head)
        self.model.requires_grad_(False)
        self._processors = {}

    def processor(self, max_pixels):
        if max_pixels not in self._processors:
            from transformers import AutoProcessor
            self._processors[max_pixels] = AutoProcessor.from_pretrained(self.ckpt, max_pixels=max_pixels)
        return self._processors[max_pixels]

    def predict(self, image, instruction, max_pixels):
        import torch
        from gui_actor.inference import inference
        proc = self.processor(max_pixels)
        with torch.no_grad():
            pred = inference(conversation(image, instruction, SYSTEM[self.backbone]), self.model, proc.tokenizer, proc,
                             logits_processor=None, use_placeholder=True, topk=3)
        pts = pred["topk_points"]
        raw = pred["output_text"] or ""
        return (None, raw) if not pts else ((float(pts[0][0]), float(pts[0][1])), raw)
