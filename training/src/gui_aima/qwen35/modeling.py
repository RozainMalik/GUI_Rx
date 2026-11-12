"""GUI-AIMA on Qwen3.5: supervise the anchor rows of the full-attention layers."""
import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import Qwen3_5ForConditionalGeneration
from transformers.modeling_outputs import ModelOutput

from gui_aima.constants import IGNORE_INDEX
from gui_aima.grounding_head import GroundingHead_MultiPatch_Attention, select_visual_sink_queries
from gui_aima.qwen35.attention import full_attention_layer_ids, recompute_attention_rows


@dataclass
class PointerOutput(ModelOutput):
    loss: torch.Tensor | None = None
    lm_loss: torch.Tensor | None = None
    pointer_loss: torch.Tensor | None = None
    pointer_scores: list | None = None
    logits: torch.Tensor | None = None


class Qwen3_5ForConditionalGenerationWithPointer(Qwen3_5ForConditionalGeneration):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.multi_patch_pointer_head_attention = GroundingHead_MultiPatch_Attention(config=self.config)
        self.pointer_loss_weight, self.lm_loss_weight = 1.0, 1.0
        self.post_init()

    def reset_loss_weights(self, pointer_loss_weight, lm_loss_weight):
        self.pointer_loss_weight, self.lm_loss_weight = pointer_loss_weight, lm_loss_weight

    def set_attention_args(self, weighting):
        if not re.fullmatch(r"query_\d+(_layer_wise)?", weighting):
            raise NotImplementedError(f"weighting={weighting!r} is not supported on Qwen3.5")
        self.config.kl_query_weighting = False
        self.config.part_query_weighting = False
        self.config.query_topk = int(re.search(r"(?<=query_)\d+", weighting).group())
        self.config.layer_wise_query_weighting = weighting.endswith("layer_wise")
        self.multi_patch_pointer_head_attention.config = self.config

    def _lm_loss(self, hidden, labels):
        """Cross-entropy on labelled positions only (never materialises full-vocab logits)."""
        shift = labels[:, 1:]
        valid = shift != IGNORE_INDEX
        if not valid.any():
            return hidden.sum() * 0.0
        logits = self.lm_head(hidden[:, :-1][valid]).float()
        return F.cross_entropy(logits, shift[valid])

    def _pointer_for_sample(self, i, input_ids, hidden_states, position_ids, attention_mask, labels):
        ids = input_ids[i]
        hs = tuple(h[i:i + 1] for h in hidden_states)
        pos = None if position_ids is None else position_ids[:, i:i + 1]
        mask = None if attention_mask is None else attention_mask[i:i + 1]
        visual = torch.nonzero(ids == self.config.image_token_id).squeeze(-1)
        targets = torch.nonzero(ids == self.config.pointer_pad_token_id).squeeze(-1)
        starts = torch.nonzero(ids == self.config.pointer_start_token_id).squeeze(-1)
        dummy = targets.numel() == 0 or starts.numel() == 0
        if dummy:  # keep the graph connected (DDP), contribute nothing
            last = int(mask[0].sum()) - 1 if mask is not None else ids.shape[0] - 1
            targets = torch.tensor([last], device=ids.device)
            query = torch.arange(visual[-1] + 1, last, device=ids.device)
            labels = torch.zeros(1, visual.numel(), device=ids.device)
            labels[0, :4] = 1
        else:
            query = torch.arange(visual[-1] + 1, starts[0], device=ids.device)
        full_ids = full_attention_layer_ids(self.config.text_config)
        attns = recompute_attention_rows(self.model.language_model, hs, pos, torch.cat([query, targets]),
                                         attention_mask=mask, layer_ids=full_ids)
        sink = select_visual_sink_queries(hs, 0, query, visual, self.config.query_topk,
                                          layer_ids=full_ids if self.config.layer_wise_query_weighting else None)
        scores, loss = self.multi_patch_pointer_head_attention(query, visual, targets, attns, sink, None,
                                                               batch_idx=0, labels=labels)
        if loss is not None and dummy:
            loss = loss * 0.0
        return scores, loss

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, labels=None, pixel_values=None,
                image_grid_thw=None, mm_token_type_ids=None, multi_patch_labels=None,
                visual_token_indices_of_coordinates=None, coordinates=None, compute_pointer=None, **kwargs):
        if position_ids is None and pixel_values is not None:
            # computed here so the attention rebuild uses exactly the forward's rotary positions
            position_ids = self.model.compute_3d_position_ids(
                input_ids=input_ids, inputs_embeds=None, image_grid_thw=image_grid_thw,
                attention_mask=attention_mask, mm_token_type_ids=mm_token_type_ids)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                         pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                         mm_token_type_ids=mm_token_type_ids, output_hidden_states=True, use_cache=False)
        hidden = out.last_hidden_state

        lm_loss = self._lm_loss(hidden, labels) if labels is not None and self.lm_loss_weight > 0 else None
        want_pointer = compute_pointer if compute_pointer is not None else multi_patch_labels is not None
        pointer_loss, pointer_scores = None, []
        if want_pointer:
            losses = []
            for i in range(input_ids.shape[0]):
                lab = multi_patch_labels[i].to(hidden.device) if multi_patch_labels is not None else None
                scores, loss = self._pointer_for_sample(i, input_ids, out.hidden_states, position_ids,
                                                        attention_mask, lab)
                pointer_scores.append(scores.detach().float().cpu())
                if loss is not None:
                    losses.append(loss)
            pointer_loss = torch.stack(losses).mean() if losses else None

        parts = []
        if lm_loss is not None:
            parts.append(self.lm_loss_weight * lm_loss)
        if pointer_loss is not None:
            parts.append(self.pointer_loss_weight * pointer_loss)
        loss = sum(parts) if parts else None
        logits = self.lm_head(hidden) if labels is None else None
        return PointerOutput(loss=loss, lm_loss=lm_loss, pointer_loss=pointer_loss,
                             pointer_scores=pointer_scores, logits=logits)
