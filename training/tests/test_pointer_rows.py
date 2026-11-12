import torch
from conftest import tiny_config
from attn_ft.freeze import cast_trainable_to_fp32, setup_full_attention_trainable
from attn_ft.pointer_rows import PointerRowEmbedding
from gui_aima.qwen35.modeling import Qwen3_5ForConditionalGenerationWithPointer
from test_modeling_qwen35 import PAD, PEND, PSTART, _batch

IDS = [PSTART, PAD, PEND]


def _model(dtype=torch.float32):
    torch.manual_seed(0)
    m = Qwen3_5ForConditionalGenerationWithPointer(tiny_config())
    m.config.pointer_start_token_id, m.config.pointer_pad_token_id, m.config.pointer_end_token_id = PSTART, PAD, PEND
    m.set_attention_args("query_1")
    return m.to(dtype)


def test_identity_at_init():
    m = _model()
    ids = torch.tensor([[1, PSTART, 5, PAD, PEND, 7]])
    before = m.get_input_embeddings()(ids).clone()
    PointerRowEmbedding.install(m, IDS)
    assert isinstance(m.get_input_embeddings(), PointerRowEmbedding)
    assert torch.equal(m.get_input_embeddings()(ids), before)


def test_only_rows_get_gradient():
    m = _model().train()
    setup_full_attention_trainable(m)
    emb = PointerRowEmbedding.install(m, IDS)
    m(**_batch(m, [24])).loss.backward()
    assert emb.weight.grad is None and not emb.weight.requires_grad
    assert emb.rows.grad is not None and emb.rows.grad.abs().sum() > 0
    assert m.model.language_model.layers[3].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert m.model.language_model.layers[4].linear_attn.in_proj_qkv.weight.grad is None


def test_state_dict_keys_and_roundtrip():
    m = _model()
    emb = PointerRowEmbedding.install(m, IDS)
    keys = [k for k in m.state_dict() if "embed_tokens" in k]
    assert sorted(k.split("embed_tokens.")[1] for k in keys) == ["rows", "token_ids", "weight"]
    with torch.no_grad():
        emb.rows.add_(1.0)
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    m2 = _model()
    PointerRowEmbedding.install(m2, IDS)
    m2.load_state_dict(sd)
    assert torch.equal(m2.get_input_embeddings().rows, emb.rows)


def test_fold_writes_rows_back():
    m = _model()
    emb = PointerRowEmbedding.install(m, IDS)
    with torch.no_grad():
        emb.rows.normal_()
    ids = torch.tensor([[1, PSTART, PAD, PEND]])
    wrapped = m.get_input_embeddings()(ids).clone()
    PointerRowEmbedding.fold(m)
    assert type(m.get_input_embeddings()) is torch.nn.Embedding
    assert torch.equal(m.get_input_embeddings()(ids), wrapped)


def test_mixed_precision_forward_backward_under_autocast():
    """GPU layout on CPU: frozen bf16, trainable fp32, bf16 autocast."""
    m = _model(torch.bfloat16).train()
    setup_full_attention_trainable(m)
    cast_trainable_to_fp32(m)
    emb = PointerRowEmbedding.install(m, IDS)
    batch = _batch(m, [24])
    batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(**batch)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert emb.rows.grad is not None and torch.isfinite(emb.rows.grad).all()
    q = m.model.language_model.layers[7].self_attn.q_proj.weight
    assert q.dtype == torch.float32 and q.grad is not None and torch.isfinite(q.grad).all()
