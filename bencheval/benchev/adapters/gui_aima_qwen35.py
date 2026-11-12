"""Our Qwen3.5 GUI-AIMA checkpoints via the vendored port (vendor/gui_aima, same code as training/src/gui_aima)."""
from benchev.adapters.base import Adapter


class GuiAimaQwen35(Adapter):
    name = "gui_aima_qwen35"
    crop_multiple = 32

    def load(self, model_cfg, device):
        from gui_aima.qwen35.inference import load_pointer_model
        self.model, self.processor = load_pointer_model(model_cfg.ckpt, model_cfg.base, device)

    def predict(self, image, instruction, max_pixels):
        from gui_aima.qwen35.inference import ground
        p = ground(self.model, self.processor, image, instruction, max_pixels)["point"]
        return (float(p[0]), float(p[1])), ""

    def predict_map(self, image, instruction, max_pixels):
        """The official click and the patch scores as a (gh, gw) float32 map (the input of zoom + fusion)."""
        from gui_aima.qwen35.inference import ground
        o = ground(self.model, self.processor, image, instruction, max_pixels)
        gh, gw = o["grid_hw"]
        m = o["scores"][0].float().cpu().numpy().reshape(gh, gw)
        p = o["point"]
        return (float(p[0]), float(p[1])), m
