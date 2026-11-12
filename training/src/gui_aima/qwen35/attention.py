"""Rebuild selected rows of Qwen3.5 full-attention maps from the saved layer inputs.

Qwen3.5 interleaves Gated-DeltaNet (linear) layers, which have no attention map, with softmax
attention layers. GUI-AIMA supervises the anchor/query rows of those maps, so we recompute only
those rows (O(R*N)) and leave the main forward on sdpa/flash.
"""
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv


def full_attention_layer_ids(text_config) -> list[int]:
    return [i for i, t in enumerate(text_config.layer_types) if t == "full_attention"]


def rope_position_ids(position_ids, batch_size, seq_len, device):
    """(3, B, N) t/h/w positions, matching what the text model feeds its rotary embedding."""
    if position_ids is None:
        return torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, batch_size, -1)
    if position_ids.ndim == 2:
        return position_ids[None].expand(3, -1, -1)
    if position_ids.shape[0] == 4:  # leading text row is only used for masking
        return position_ids[1:]
    return position_ids


def recompute_attention_rows(language_model, hidden_states, position_ids, row_indices,
                             attention_mask=None, layer_ids=None, return_gates=False):
    """Attention probabilities of `row_indices` over all keys, for each full-attention layer.

    hidden_states: tuple from output_hidden_states=True; hidden_states[i] is the input of layer i.
    Returns a list of (B, H, R, N) float32 tensors (plus (B, H, R) mean output gates if requested).
    """
    cfg = language_model.config
    layer_ids = full_attention_layer_ids(cfg) if layer_ids is None else layer_ids
    x0 = hidden_states[0]
    bsz, seq_len, _ = x0.shape
    rows = torch.as_tensor(row_indices, device=x0.device, dtype=torch.long)
    cos, sin = language_model.rotary_emb(x0, rope_position_ids(position_ids, bsz, seq_len, x0.device))

    masked = torch.arange(seq_len, device=x0.device)[None, :] > rows[:, None]      # (R, N) future keys
    masked = masked[None, None].expand(bsz, 1, -1, -1)
    if attention_mask is not None:
        masked = masked | (attention_mask[:, None, None, :] == 0)

    attns, gates = [], []
    for i in layer_ids:
        layer = language_model.layers[i]
        attn = layer.self_attn
        h = layer.input_layernorm(hidden_states[i])
        hd = attn.head_dim
        q, gate = torch.chunk(attn.q_proj(h[:, rows]).view(bsz, len(rows), -1, hd * 2), 2, dim=-1)
        q = attn.q_norm(q).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(h).view(bsz, seq_len, -1, hd)).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, q, cos[:, rows], sin[:, rows])
        k, _ = apply_rotary_pos_emb(k, k, cos, sin)
        k = repeat_kv(k, attn.num_key_value_groups)
        scores = torch.matmul(q, k.transpose(2, 3)).float() * attn.scaling
        scores = scores.masked_fill(masked, torch.finfo(scores.dtype).min)
        attns.append(F.softmax(scores, dim=-1))
        if return_gates:
            gates.append(torch.sigmoid(gate.float()).mean(-1).transpose(1, 2))
    return (attns, gates) if return_gates else attns
