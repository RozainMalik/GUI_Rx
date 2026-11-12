import pytest
import torch
from conftest import make_mm_inputs, tiny_config
from gui_aima.qwen35.modeling import Qwen3_5ForConditionalGenerationWithPointer

PSTART, PAD, PEND = 310, 311, 312


def _model():
    torch.manual_seed(0)
    m = Qwen3_5ForConditionalGenerationWithPointer(tiny_config())
    m.config.pointer_start_token_id, m.config.pointer_pad_token_id, m.config.pointer_end_token_id = PSTART, PAD, PEND
    m.set_attention_args("query_1")
    return m.float()


def _batch(m, lengths, with_pointer=True):
    x = make_mm_inputs(m.config, lengths)
    labels = torch.full_like(x["input_ids"], -100)
    mpl = []
    for b, n in enumerate(lengths):
        if with_pointer:
            x["input_ids"][b, n - 4:n - 1] = torch.tensor([PSTART, PAD, PEND])
        labels[b, n - 5:n] = x["input_ids"][b, n - 5:n]
        lab = torch.zeros(4); lab[1] = 1.0
        mpl.append(lab[None])
    x["mm_token_type_ids"] = (x["input_ids"] == 300).int()
    return dict(**x, labels=labels, multi_patch_labels=mpl,
                visual_token_indices_of_coordinates=[torch.tensor([1])] * len(lengths))


def test_forward_losses_finite_and_scores_normalised():
    m = _model().train()
    out = m(**_batch(m, [24, 20]))
    assert torch.isfinite(out.loss) and torch.isfinite(out.pointer_loss) and torch.isfinite(out.lm_loss)
    assert len(out.pointer_scores) == 2
    for s in out.pointer_scores:
        assert s.shape == (1, 4) and abs(float(s.sum()) - 1) < 1e-4


def test_pointer_loss_trains_full_attention_qk():
    m = _model().train()
    m.reset_loss_weights(pointer_loss_weight=1.0, lm_loss_weight=0.0)
    m(**_batch(m, [24])).loss.backward()
    lm = m.model.language_model
    assert lm.layers[3].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert lm.layers[7].self_attn.k_proj.weight.grad.abs().sum() > 0


def test_missing_pointer_gives_zero_loss():
    m = _model().train()
    m.reset_loss_weights(pointer_loss_weight=1.0, lm_loss_weight=0.0)
    out = m(**_batch(m, [24], with_pointer=False))
    assert out.pointer_loss.item() == 0.0
    out.loss.backward()                          # still in the graph (DDP needs this)


def test_unsupported_weighting_raises():
    m = _model()
    for w in ["query_kl_part", "kl", "query_1_part"]:
        with pytest.raises(NotImplementedError):
            m.set_attention_args(w)


def test_layer_wise_runs():
    m = _model().train()
    m.set_attention_args("query_1_layer_wise")
    assert torch.isfinite(m(**_batch(m, [24])).loss)


def test_lm_loss_matches_full_logits_reference():
    m = _model().eval()
    b = _batch(m, [24, 20])
    with torch.no_grad():
        out = m(**b)
        ref = torch.nn.functional.cross_entropy(
            m(**{k: v for k, v in b.items() if k != "labels"}, compute_pointer=False).logits[:, :-1].reshape(-1, 320).float(),
            b["labels"][:, 1:].reshape(-1), ignore_index=-100)
    torch.testing.assert_close(out.lm_loss, ref)


def test_ground_returns_normalised_point():
    import os, subprocess, sys
    from PIL import Image
    from transformers import AutoProcessor, AutoTokenizer
    from gui_aima.constants import (ADDITIONAL_SPECIAL_TOKENS, DEFAULT_POINTER_END_TOKEN,
                                    DEFAULT_POINTER_PAD_TOKEN, DEFAULT_POINTER_START_TOKEN)
    from gui_aima.qwen35.inference import ground, ground_two_step
    tiny = "tests/_artifacts/tiny_qwen35"
    if not os.path.exists(os.path.join(tiny, "config.json")):
        subprocess.run([sys.executable, "scripts/make_tiny_qwen35.py", "--out", tiny], check=True)
    tok = AutoTokenizer.from_pretrained(tiny)
    tok.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    proc = AutoProcessor.from_pretrained(tiny)
    proc.tokenizer = tok
    m = Qwen3_5ForConditionalGenerationWithPointer.from_pretrained(tiny).float().eval()
    m.resize_token_embeddings(len(tok))
    m.config.pointer_start_token_id = tok.convert_tokens_to_ids(DEFAULT_POINTER_START_TOKEN)
    m.config.pointer_pad_token_id = tok.convert_tokens_to_ids(DEFAULT_POINTER_PAD_TOKEN)
    m.config.pointer_end_token_id = tok.convert_tokens_to_ids(DEFAULT_POINTER_END_TOKEN)
    m.set_attention_args("query_1")
    img = Image.new("RGB", (1366, 768), "white")
    r = ground(m, proc, img, "Click save.", max_pixels=200 * 32 * 32)
    assert 0 <= r["point"][0] <= 1 and 0 <= r["point"][1] <= 1
    assert r["scores"].shape[-1] == r["grid_hw"][0] * r["grid_hw"][1]
    r2 = ground_two_step(m, proc, img, "Click save.", max_pixels=200 * 32 * 32)
    assert 0 <= r2["point"][0] <= 1 and 0 <= r2["point"][1] <= 1
