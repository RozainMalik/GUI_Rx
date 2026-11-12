"""Save a tiny random Qwen3.5 (real tokenizer/processor, real special-token ids) for smoke tests."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from conftest import tiny_config  # noqa: E402
from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="tests/_artifacts/tiny_qwen35")
ap.add_argument("--src", default="Qwen/Qwen3.5-0.8B")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.src)
cfg = tiny_config(vocab_size=len(tok))
cfg.image_token_id = tok.convert_tokens_to_ids("<|image_pad|>")
cfg.video_token_id = tok.convert_tokens_to_ids("<|video_pad|>")
cfg.vision_start_token_id = tok.convert_tokens_to_ids("<|vision_start|>")
cfg.vision_end_token_id = tok.convert_tokens_to_ids("<|vision_end|>")
Qwen3_5ForConditionalGeneration(cfg).save_pretrained(a.out)
AutoProcessor.from_pretrained(a.src).save_pretrained(a.out)
tok.save_pretrained(a.out)
print("saved", a.out)
