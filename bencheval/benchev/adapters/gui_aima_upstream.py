"""Published Qwen2.5-VL GUI-AIMA models via the official inference code (vendor/upstream_gui_aima, sjz5202/GUI-AIMA@a0b938e).

Each call mirrors one sample of upstream eval/screenSpot_pro.py: the same system message, `inference(...,
use_placeholder=True, topk=3)`, the top-1 region point, and an AutoProcessor built with `max_pixels` (one per value).
Needs the upstream venv (torch 2.6, transformers 4.51.3, flash-attn 2.7.3), not the Qwen3.5 one.
"""
import sys

from benchev import ROOT
from benchev.adapters.base import Adapter

UPSTREAM = str(ROOT / "vendor" / "upstream_gui_aima")
SYSTEM = ("You are a GUI agent. Given a screenshot of the current GUI and a human instruction, your task is to locate "
          "the screen element that corresponds to the instruction. You should output a PyAutoGUI action that performs "
          "a click on the correct position. To indicate the click location, we will use some special tokens, which is "
          "used to refer to a visual patch later. For example, you can output: pyautogui.click(<your_special_token_here>).")


def use_upstream_package():
    """Make `import gui_aima` resolve to the upstream package, not our Qwen3.5 port in vendor/gui_aima."""
    for m in [m for m in sys.modules if m == "gui_aima" or m.startswith("gui_aima.")]:
        del sys.modules[m]
    if UPSTREAM in sys.path:
        sys.path.remove(UPSTREAM)
    sys.path.insert(0, UPSTREAM)


def last_position_head(lm_head):
    """lm_head that projects only the last position. Upstream's forward computes logits for the whole prompt
    (~7k image tokens x 151k vocab = 2.1 GB bf16), which does not fit a 12 GB GPU. Generation reads only the last
    position's logits and the pointer comes from hidden states, so predictions are unchanged."""
    import torch.nn as nn

    class LastPositionHead(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        @property
        def weight(self):
            return self.inner.weight

        def forward(self, hidden_states):
            return self.inner(hidden_states[:, -1:, :])

    return LastPositionHead(lm_head)


def conversation(image, instruction):
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}]}]


class GuiAimaUpstream(Adapter):
    name = "gui_aima_upstream"
    crop_multiple = 28

    def load(self, model_cfg, device):
        use_upstream_package()
        import torch
        from gui_aima.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
        self.ckpt = model_cfg.ckpt
        self.model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
            self.ckpt, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="flash_attention_2").eval()
        self.model.lm_head = last_position_head(self.model.lm_head)
        # upstream inference() recomputes Q/K attention for all layers outside no_grad; with trainable weights autograd
        # keeps every layer's intermediates (~3 GB at 5 MP). Inference only, so gradients off: same numbers, less memory.
        self.model.requires_grad_(False)
        self._processors = {}

    def processor(self, max_pixels):
        if max_pixels not in self._processors:
            from transformers import AutoProcessor
            self._processors[max_pixels] = AutoProcessor.from_pretrained(self.ckpt, max_pixels=max_pixels)
        return self._processors[max_pixels]

    def predict(self, image, instruction, max_pixels):
        import torch
        from gui_aima.inference import inference
        proc = self.processor(max_pixels)
        with torch.no_grad():
            pred = inference(conversation(image, instruction), self.model, proc.tokenizer, proc,
                             logits_processor=None, use_placeholder=True, topk=3)
        pts = pred["topk_points"]
        raw = pred["output_text"] or ""
        return (None, raw) if not pts else ((float(pts[0][0]), float(pts[0][1])), raw)
