"""Which weights train: the attention blocks of Qwen3.5's full-attention layers (plus the pointer rows, see pointer_rows.py).

Only these layers produce the attention maps GUI-AIMA supervises; everything else stays frozen in bf16.
"""
import re

import torch

from gui_aima.qwen35.attention import full_attention_layer_ids

TRAINABLE_SUBMODULES = ("self_attn", "input_layernorm")
ALLOWED_PATTERN = re.compile(r"language_model\.layers\.(\d+)\.(self_attn|input_layernorm)\.|embed_tokens\.rows$")


def is_allowed(name: str, full_ids: list[int]) -> bool:
    m = ALLOWED_PATTERN.search(name)
    if not m:
        return False
    return m.group(1) is None or int(m.group(1)) in full_ids


def _trainable_modules(model):
    lm = model.model.language_model
    ids = full_attention_layer_ids(lm.config)
    if not ids:
        raise ValueError("model has no full_attention layers (text_config.layer_types)")
    return [getattr(lm.layers[i], sub) for i in ids for sub in TRAINABLE_SUBMODULES]


def setup_full_attention_trainable(model) -> list[str]:
    for p in model.parameters():
        p.requires_grad = False
    for mod in _trainable_modules(model):
        for p in mod.parameters():
            p.requires_grad = True
    names = sorted(n for n, p in model.named_parameters() if p.requires_grad)
    if not names:
        raise ValueError("no trainable parameters after freezing")
    return names


def cast_trainable_to_fp32(model) -> None:
    """fp32 master weights for the trained modules only; --bf16 autocast still runs their matmuls in bf16."""
    for mod in _trainable_modules(model):
        mod.to(torch.float32)


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
