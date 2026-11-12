import torch
from gui_aima.qwen35.attention import full_attention_layer_ids, recompute_attention_rows
from conftest import make_mm_inputs


def _forward(model, x):
    pos = model.model.compute_3d_position_ids(
        input_ids=x["input_ids"], inputs_embeds=None, image_grid_thw=x["image_grid_thw"],
        attention_mask=x["attention_mask"], mm_token_type_ids=x["mm_token_type_ids"])
    out = model(**x, position_ids=pos, output_hidden_states=True, output_attentions=True)
    return out, pos


def test_full_layer_ids(tiny_model):
    assert full_attention_layer_ids(tiny_model.config.text_config) == [3, 7]


@torch.no_grad()
def test_rows_match_eager_single(tiny_model):
    x = make_mm_inputs(tiny_model.config, [20])
    out, pos = _forward(tiny_model, x)
    rows = torch.tensor([8, 12, 19])
    got = recompute_attention_rows(tiny_model.model.language_model, out.hidden_states, pos, rows,
                                   attention_mask=x["attention_mask"])
    assert len(got) == len(out.attentions) == 2
    for mine, ref in zip(got, out.attentions):
        torch.testing.assert_close(mine, ref[:, :, rows].float(), atol=1e-5, rtol=1e-4)


@torch.no_grad()
def test_rows_match_eager_with_padding(tiny_model):
    x = make_mm_inputs(tiny_model.config, [22, 15], seed=1)
    out, pos = _forward(tiny_model, x)
    rows = torch.tensor([9, 14])                       # valid for both samples
    got = recompute_attention_rows(tiny_model.model.language_model, out.hidden_states, pos, rows,
                                   attention_mask=x["attention_mask"])
    for mine, ref in zip(got, out.attentions):
        for b, n in enumerate([22, 15]):
            torch.testing.assert_close(mine[b, :, :, :n], ref[b, :, rows, :n].float(), atol=1e-5, rtol=1e-4)
            if n < mine.shape[-1]:
                assert mine[b, :, :, n:].abs().max() < 1e-6    # padded keys get no mass


@torch.no_grad()
def test_text_only_positions_none(tiny_model):
    ids = torch.randint(0, 250, (1, 12))
    out = tiny_model(input_ids=ids, output_hidden_states=True, output_attentions=True)
    got = recompute_attention_rows(tiny_model.model.language_model, out.hidden_states, None, torch.tensor([11]))
    torch.testing.assert_close(got[1], out.attentions[1][:, :, [11]].float(), atol=1e-5, rtol=1e-4)


def test_gradients_flow_to_full_layer_qk(tiny_model):
    tiny_model.train()
    x = make_mm_inputs(tiny_model.config, [20])
    out, pos = _forward(tiny_model, x)
    attns = recompute_attention_rows(tiny_model.model.language_model, out.hidden_states, pos, torch.tensor([19]))
    torch.stack([a[..., 3:7].sum() for a in attns]).sum().backward()
    lm = tiny_model.model.language_model
    assert lm.layers[3].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert lm.layers[7].self_attn.k_proj.weight.grad.abs().sum() > 0
    assert lm.layers[0].linear_attn.in_proj_qkv.weight.grad.abs().sum() > 0   # via hidden states


def test_gates_shape(tiny_model):
    x = make_mm_inputs(tiny_model.config, [20])
    with torch.no_grad():
        out, pos = _forward(tiny_model, x)
        _, gates = recompute_attention_rows(tiny_model.model.language_model, out.hidden_states, pos,
                                            torch.tensor([5, 19]), return_gates=True)
    assert [g.shape for g in gates] == [torch.Size([1, 4, 2])] * 2
    assert all(((g > 0) & (g < 1)).all() for g in gates)
