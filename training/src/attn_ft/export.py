"""Training checkpoint (wrapped embedding, fp32 trainable modules) -> plain pointer checkpoint the eval scripts load."""
import argparse
import glob
import os

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoProcessor, AutoTokenizer

from gui_aima.qwen35.modeling import Qwen3_5ForConditionalGenerationWithPointer

# Written into out_dir after everything else. save_pretrained writes config.json before the weights, so a job killed
# mid-export leaves a config.json; only this marker means the export is complete.
EXPORT_DONE = "EXPORT_DONE"


def read_pointer_rows(ckpt_dir):
    """(token_ids, rows) from the checkpoint's safetensors, matched by key suffix (save-time key renames are OK)."""
    found = {}
    for f in sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors"))):
        with safe_open(f, framework="pt") as st:
            for k in st.keys():
                for suffix in ("embed_tokens.rows", "embed_tokens.token_ids"):
                    if k.endswith(suffix):
                        found[suffix] = st.get_tensor(k)
    for suffix in ("embed_tokens.rows", "embed_tokens.token_ids"):
        if suffix not in found:
            raise KeyError(f"{suffix} not found in {ckpt_dir}/*.safetensors")
    return found["embed_tokens.token_ids"], found["embed_tokens.rows"]


def _is_tied(config) -> bool:
    return bool(getattr(config, "tie_word_embeddings", False)
                or getattr(config.get_text_config(), "tie_word_embeddings", False))


def _untie(model) -> None:
    """Give lm_head its own copy of the (frozen) table, so writing the trained input rows leaves lm_head as trained."""
    model.lm_head.weight = nn.Parameter(model.get_input_embeddings().weight.detach().clone())
    model.config.tie_word_embeddings = False
    model.config.get_text_config().tie_word_embeddings = False


def export_bf16(ckpt_dir, out_dir, dtype=torch.bfloat16) -> str:
    token_ids, rows = read_pointer_rows(ckpt_dir)
    model = Qwen3_5ForConditionalGenerationWithPointer.from_pretrained(ckpt_dir, dtype=dtype)  # ignores rows/token_ids
    if _is_tied(model.config):
        _untie(model)
    emb = model.get_input_embeddings().weight
    with torch.no_grad():
        emb[token_ids] = rows.to(dtype)
    if not torch.equal(emb[token_ids], rows.to(dtype)):
        raise RuntimeError("pointer rows were not written into the embedding")
    os.makedirs(out_dir, exist_ok=True)
    done = os.path.join(out_dir, EXPORT_DONE)
    if os.path.exists(done):
        os.remove(done)
    model.save_pretrained(out_dir)
    AutoProcessor.from_pretrained(ckpt_dir).save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(ckpt_dir).save_pretrained(out_dir)
    with open(done, "w") as fh:
        fh.write(f"exported from {ckpt_dir}\n")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    a = ap.parse_args()
    print(export_bf16(a.ckpt_dir, a.out_dir, getattr(torch, a.dtype)))
