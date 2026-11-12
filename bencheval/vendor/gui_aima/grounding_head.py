"""GUI-AIMA grounding head and visual-sink query selection, shared by all backbones."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroundingHead_MultiPatch_Attention(nn.Module):
    def __init__(self, config=None, topk=-1):
        super().__init__()
        self.config = config
        self.topk = topk
    
    # no softmax head weight first, then weighting
    def forward(self,
                query_indices,
                visual_indices,
                target_indices,
                self_attentions,
                topk_query_indices,
                global_pattern_per_query,
                batch_idx,
                labels: Optional[torch.Tensor] = None,  # shape: [n_dec, n_enc], binary mask of patches in bbox
                do_single_patch: bool = False,
               ):
        all_head_attentions = []
        query_head_attns = []
        epsilon = 1e-8
        # lay_n , batch_n, head_n, seq_n, seq_n
        for layer_idx in range(len(self_attentions)):
            layer_attn = self_attentions[layer_idx]
            sample_layer_attn = layer_attn[batch_idx]
            if self.config.kl_query_weighting or self.config.query_topk is not None:
                if not self.config.kl_query_weighting and topk_query_indices is not None:
                    if self.config.layer_wise_query_weighting:
                        q_attn = sample_layer_attn[:, topk_query_indices[layer_idx], :][:, :, visual_indices]
                    else:
                        q_attn = sample_layer_attn[:, topk_query_indices, :][:, :, visual_indices]
                elif self.config.kl_query_weighting and global_pattern_per_query is not None:
                    q_attn = sample_layer_attn[:, 0:-1, visual_indices]
                query_head_attns.append(q_attn)  
            target_patch_attentions = sample_layer_attn[:, -1, visual_indices.unsqueeze(0)]
            if target_patch_attentions.shape[1]!=1:
                target_patch_attentions=torch.mean(target_patch_attentions, dim=1,keepdim=True)
            target_patch_attentions=target_patch_attentions.squeeze(1)
            all_head_attentions.append(target_patch_attentions)
        # lay_n * head_n, 1, patch_n
        all_head_attentions_cat = torch.cat(all_head_attentions, dim=0)
        if self.config.kl_query_weighting or self.config.query_topk is not None:
            query_head_attns_cat = torch.cat(query_head_attns, dim=0) 
            if not self.config.kl_query_weighting and topk_query_indices is not None:
                head_weights = query_head_attns_cat.sum(dim=(-1, -2)) 
                head_weights = (head_weights).softmax(dim=-1)
            elif self.config.kl_query_weighting and global_pattern_per_query is not None:
                head_weights = query_head_attns_cat.sum(dim=(-1)) 
                head_weights = (head_weights).softmax(dim=-1)
                head_weights = head_weights.clamp_min(1e-12)               
                global_pattern_per_query = global_pattern_per_query.clamp_min(1e-12)
                distance =- F.kl_div(
                    (head_weights).log(),                      
                    global_pattern_per_query.expand_as(head_weights),                 
                    reduction='none').sum(dim=-1) 
                head_weights=(distance).softmax(dim=-1)  

        if self.topk != -1:
            with torch.no_grad():
            # lay_n * head_n, 1
                flat_head_weights=all_head_attentions_cat.sum(dim=-1)
            _, topk_idx = torch.topk(flat_head_weights, self.topk, largest=True)
            target_patch_attentions_topk = all_head_attentions_cat[topk_idx]
        else:
            target_patch_attentions_topk = all_head_attentions_cat

        if self.training:
            del all_head_attentions_cat, all_head_attentions
            if topk_query_indices is not None:
                del query_head_attns_cat
        else:
            del all_head_attentions_cat, all_head_attentions
            if topk_query_indices is not None:
                del query_head_attns_cat
            torch.cuda.empty_cache()

        if self.config.kl_query_weighting or self.config.query_topk is not None:
            target_patch_attentions_merged = torch.mul(head_weights[:, None], target_patch_attentions_topk)
            target_patch_attentions_merged = target_patch_attentions_merged.sum(dim=0,keepdim=True)
        else:
            target_patch_attentions_merged = target_patch_attentions_topk.mean(dim=0,keepdim=True)
        
        target_patch_attentions_merged /= (target_patch_attentions_merged.sum(dim=-1, keepdim=True) + epsilon)    
        attn_weights = target_patch_attentions_merged

        loss = None
        if (labels is not None) and (not do_single_patch):
            labels_float = labels.float()
            # Normalize each row to get target probability distribution
            target_dist = labels_float / (labels_float.sum(dim=-1, keepdim=True) + epsilon)

            pred_log_probs = torch.log(attn_weights)
            # Use KL divergence as loss
            loss = F.kl_div(pred_log_probs, target_dist, reduction='batchmean')

        return attn_weights, loss


@torch.no_grad()
def select_visual_sink_queries(hidden_states, batch_idx, query_indices, visual_indices, k, layer_ids=None):
    """GUI-AIMA Eq. 4-5: query tokens whose hidden states align most with the visual tokens.

    Returns local indices into `query_indices`: (k,) summed over all layer outputs, or
    (len(layer_ids), k) chosen from the inputs of the given layers (layer-wise mode).
    """
    def affinity(h):
        q = F.normalize(h[batch_idx, query_indices].float(), dim=-1)
        v = F.normalize(h[batch_idx, visual_indices].float(), dim=-1)
        return (q @ v.T).sum(-1)                                   # (Q,)

    if layer_ids is not None:
        return torch.stack([torch.topk(affinity(hidden_states[i]), k).indices for i in layer_ids])
    total = sum(affinity(h) for h in hidden_states[1:])
    return torch.topk(total, k).indices
