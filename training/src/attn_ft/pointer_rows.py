"""Train only the pointer-token rows of the (frozen, bf16) input embedding.

Unfreezing the whole 248k x 4096 table would cost ~16 GB of fp32 masters, grads and
Adam states on the 9B for rows that never change. The wrapper keeps the table frozen and substitutes a small fp32
parameter for the new-token rows. The state-dict keys stay `embed_tokens.weight` (+ `rows`, `token_ids`).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerRowEmbedding(nn.Module):
    def __init__(self, base: nn.Embedding, token_ids):
        super().__init__()
        self.weight = base.weight
        self.weight.requires_grad = False
        self.padding_idx = base.padding_idx
        self.num_embeddings, self.embedding_dim = base.weight.shape
        ids = torch.as_tensor(list(token_ids), dtype=torch.long, device=base.weight.device)
        self.register_buffer("token_ids", ids, persistent=True)
        self.rows = nn.Parameter(base.weight.detach()[ids].clone().float(), requires_grad=True)

    def forward(self, input_ids):
        out = F.embedding(input_ids, self.weight, self.padding_idx)
        hit = input_ids.unsqueeze(-1) == self.token_ids                       # (..., n)
        sub = hit.to(self.rows.dtype) @ self.rows                              # (..., hidden)
        return torch.where(hit.any(-1, keepdim=True), sub.to(out.dtype), out)

    @staticmethod
    def install(model, token_ids) -> "PointerRowEmbedding":
        emb = PointerRowEmbedding(model.get_input_embeddings(), token_ids)
        model.set_input_embeddings(emb)
        return emb

    @staticmethod
    def fold(model) -> None:
        emb = model.get_input_embeddings()
        if not isinstance(emb, PointerRowEmbedding):
            return
        plain = nn.Embedding(emb.num_embeddings, emb.embedding_dim, padding_idx=emb.padding_idx,
                             device=emb.weight.device, dtype=emb.weight.dtype)
        with torch.no_grad():
            plain.weight.copy_(emb.weight)
            plain.weight[emb.token_ids] = emb.rows.to(plain.weight.dtype)
        model.set_input_embeddings(plain)
