import pytest
import torch
from conftest import tiny_config
from attn_ft.freeze import (cast_trainable_to_fp32, count_trainable, is_allowed,
                            setup_full_attention_trainable)
from gui_aima.qwen35.modeling import Qwen3_5ForConditionalGenerationWithPointer


def _model():
    torch.manual_seed(0)
    return Qwen3_5ForConditionalGenerationWithPointer(tiny_config()).to(torch.bfloat16)


def test_trainable_set_is_exactly_full_attention_blocks():
    m = _model()
    names = setup_full_attention_trainable(m)
    assert names, "nothing trainable"
    for n in names:
        assert ".layers.3." in n or ".layers.7." in n, n
        assert ".self_attn." in n or ".input_layernorm." in n, n
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable == set(names)
    expected = sum(p.numel() for n, p in m.named_parameters()
                   if (".layers.3." in n or ".layers.7." in n) and (".self_attn." in n or ".input_layernorm." in n)
                   and "language_model" in n)
    assert count_trainable(m) == expected
    for n, p in m.named_parameters():
        if any(k in n for k in ("visual", "mlp", "linear_attn", "lm_head", "embed_tokens", "post_attention_layernorm")):
            assert not p.requires_grad, n


def test_cast_only_trainable_to_fp32():
    m = _model()
    setup_full_attention_trainable(m)
    cast_trainable_to_fp32(m)
    for n, p in m.named_parameters():
        assert p.dtype == (torch.float32 if p.requires_grad else torch.bfloat16), n


def test_no_full_attention_layers_raises():
    m = _model()
    m.config.text_config.layer_types = ["linear_attention"] * len(m.config.text_config.layer_types)
    with pytest.raises(ValueError, match="no full_attention layers"):
        setup_full_attention_trainable(m)


def test_is_allowed():
    assert is_allowed("model.language_model.layers.3.self_attn.q_proj.weight", [3, 7])
    assert is_allowed("model.language_model.layers.7.input_layernorm.weight", [3, 7])
    assert is_allowed("model.language_model.embed_tokens.rows", [3, 7])
    assert not is_allowed("model.language_model.layers.4.self_attn.q_proj.weight", [3, 7])
    assert not is_allowed("model.language_model.layers.3.mlp.up_proj.weight", [3, 7])
    assert not is_allowed("model.language_model.layers.13.self_attn.q_proj.weight", [3, 7])
