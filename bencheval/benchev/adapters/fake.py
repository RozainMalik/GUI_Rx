from benchev.adapters.base import Adapter


class FakeAdapter(Adapter):
    """Deterministic adapter for tests: fn(image, instruction) -> point | None. Records each call in .calls."""
    name = "fake"

    def __init__(self, fn, crop_multiple=28, map_fn=None):
        self.fn, self.crop_multiple, self.calls, self.map_fn = fn, crop_multiple, [], map_fn

    def load(self, model_cfg, device):
        pass

    def predict(self, image, instruction, max_pixels):
        self.calls.append({"size": image.size, "max_pixels": max_pixels})
        return self.fn(image, instruction), ""

    def predict_map(self, image, instruction, max_pixels):
        """map_fn(image, instruction) -> (gh, gw) array or None; the point is fn's."""
        self.calls.append({"size": image.size, "max_pixels": max_pixels})
        return self.fn(image, instruction), self.map_fn(image, instruction)
