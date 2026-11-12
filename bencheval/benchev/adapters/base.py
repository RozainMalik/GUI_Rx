from __future__ import annotations

from abc import ABC, abstractmethod


class Adapter(ABC):
    name: str = "base"
    crop_multiple: int = 28     # visual token side in px (Qwen2.5-VL 28, Qwen3.5 32)

    @abstractmethod
    def load(self, model_cfg, device: str) -> None: ...

    @abstractmethod
    def predict(self, image, instruction: str, max_pixels: int):
        """Return (point normalised to `image` or None, raw text)."""

    def predict_map(self, image, instruction: str, max_pixels: int):
        """Return (point normalised to `image` or None, the (gh, gw) grounding map over `image` as a numpy array or
        None). Only attention grounders have a map; the zoom + fusion mode ("zf") needs it."""
        raise NotImplementedError(f"{self.name}: no grounding map, so no zoom + fusion")


class RemoteRecipeAdapter(Adapter):
    """A model with its own inference code and fixed image handling: one user text from the recipe, the answer parsed
    and mapped with the given image's size (these recipes use 0-999 / 0-1000 / [0, 1] coordinates)."""
    crop_multiple = 28      # unused: comparison models run 1-step only

    def _generate(self, image, text: str) -> str: ...

    def predict(self, image, instruction, max_pixels):
        from benchev.adapters.recipes import parse, render, to_norm
        _, text = render(self.recipe, instruction, image.size, image.size)
        raw = self._generate(image, text)
        pt = parse(self.recipe.parser, raw)
        return (None if pt is None else to_norm(pt, self.recipe.space, image.size, image.size)), raw
