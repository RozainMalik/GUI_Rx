"""Zero-shot base Qwen3.5 (2B/4B/9B) that writes the click point as text, via the vendored
gui_aima.qwen35.text_coord.

Prompt "tool": the chat template's own computer_use tool, coordinates 0-1000 (Qwen's GUI-agent format; it beat the
plain point_2d prompt on ScreenSpot-v2). Greedy decoding,
thinking off, 256 new tokens. An unparsable answer is a miss (point None); the raw text is kept in the predictions.
"""
from benchev.adapters.base import Adapter

PROMPT = "tool"


class TextCoords(Adapter):
    name = "text_coords"
    crop_multiple = 32      # Qwen3.5: 16 px patches, 2x2 merge

    def load(self, model_cfg, device):
        from gui_aima.qwen35.text_coord import load_base_model
        self.model, self.processor = load_base_model(model_cfg.ckpt, device)

    def predict(self, image, instruction, max_pixels):
        from gui_aima.qwen35.text_coord import ground_text
        o = ground_text(self.model, self.processor, image, instruction, max_pixels, prompt=PROMPT)
        p = o["point"]
        return (None if p is None else (float(p[0]), float(p[1]))), o["raw"]
