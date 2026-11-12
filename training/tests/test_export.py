import json
import os
import subprocess
import sys

import pytest
import torch
from transformers import AutoProcessor, AutoTokenizer

from attn_ft.export import EXPORT_DONE, export_bf16, read_pointer_rows
from attn_ft.pointer_rows import PointerRowEmbedding
from gui_aima.constants import ADDITIONAL_SPECIAL_TOKENS
from gui_aima.qwen35.inference import load_pointer_model
from gui_aima.qwen35.modeling import Qwen3_5ForConditionalGenerationWithPointer
from test_dataset_qwen35 import tiny_dir  # noqa: F401


@pytest.fixture
def training_ckpt(tiny_dir, tmp_path):
    """A checkpoint shaped like the Trainer's: wrapped embedding with trained rows, plus tokenizer and processor."""
    tok = AutoTokenizer.from_pretrained(tiny_dir)
    tok.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    proc = AutoProcessor.from_pretrained(tiny_dir)
    proc.tokenizer = tok
    torch.manual_seed(0)
    m = Qwen3_5ForConditionalGenerationWithPointer.from_pretrained(tiny_dir, dtype=torch.float32)
    m.resize_token_embeddings(len(tok))
    ids = tok.convert_tokens_to_ids(ADDITIONAL_SPECIAL_TOKENS)
    emb = PointerRowEmbedding.install(m, ids)
    with torch.no_grad():
        emb.rows.normal_()
    out = tmp_path / "checkpoint-2"
    m.save_pretrained(out)
    proc.save_pretrained(out)
    tok.save_pretrained(out)
    return str(out), m, ids, tok


def test_read_pointer_rows_by_suffix(training_ckpt):
    ckpt, m, ids, _ = training_ckpt
    tids, rows = read_pointer_rows(ckpt)
    assert tids.tolist() == ids
    assert torch.equal(rows, m.get_input_embeddings().rows.detach())


def test_export_matches_wrapped_model(training_ckpt, tmp_path):
    ckpt, m, ids, tok = training_ckpt
    out = export_bf16(ckpt, str(tmp_path / "exp"), dtype=torch.float32)
    ref = m.eval()
    ref.config.pointer_start_token_id, ref.config.pointer_pad_token_id, ref.config.pointer_end_token_id = ids[2], ids[4], ids[3]
    x = torch.tensor([[1, 2, ids[2], ids[4], ids[3], 5]])
    with torch.no_grad():
        want = ref(input_ids=x).logits
    loaded, _ = load_pointer_model(out, device="cpu")
    with torch.no_grad():
        got = loaded(input_ids=x).logits
    assert torch.allclose(got, want, atol=1e-5)
    assert type(loaded.get_input_embeddings()) is torch.nn.Embedding
    assert os.path.exists(os.path.join(out, EXPORT_DONE))


def test_interrupted_export_leaves_no_done_marker(training_ckpt, tmp_path, monkeypatch):
    """config.json is written before the weights and tokenizer, so only EXPORT_DONE (written last) means complete;
    a re-export first removes the old marker."""
    ckpt, *_ = training_ckpt
    out = tmp_path / "exp"
    out.mkdir()
    (out / EXPORT_DONE).write_text("old")

    def boom(*a, **k):
        raise RuntimeError("killed mid-export")
    monkeypatch.setattr("attn_ft.export.AutoTokenizer.from_pretrained", boom)
    with pytest.raises(RuntimeError):
        export_bf16(ckpt, str(out), dtype=torch.float32)
    assert (out / "config.json").exists() and not (out / EXPORT_DONE).exists()


def test_export_bf16_default_dtype(training_ckpt, tmp_path):
    ckpt, _, _, _ = training_ckpt
    out = export_bf16(ckpt, str(tmp_path / "exp16"))
    cfg = json.load(open(os.path.join(out, "config.json")))
    assert "bfloat16" in json.dumps(cfg)
    assert os.path.exists(os.path.join(out, "tokenizer_config.json"))


def test_missing_rows_raises(tiny_dir):
    with pytest.raises(KeyError, match="embed_tokens.rows"):
        read_pointer_rows(tiny_dir)


def test_exported_ckpt_runs_through_eval_script(training_ckpt, tmp_path):
    from PIL import Image
    ckpt, _, _, _ = training_ckpt
    out = export_bf16(ckpt, str(tmp_path / "exp_eval"), dtype=torch.float32)
    data = tmp_path / "ssv2"
    data.mkdir()
    Image.new("RGB", (640, 360), "white").save(data / "a.png")
    json.dump([{"img_filename": "a.png", "bbox": [0, 0, 640, 360], "instruction": "Click.", "split": "web",
                "data_type": "text"}], open(data / "subset.json", "w"))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(["src", "tests", "."]))
    subprocess.run([sys.executable, "eval/screenspot_v2_qwen35.py", "--device", "cpu", "--ckpt", out, "--data", str(data),
                    "--out", str(tmp_path / "ev"), "--max_pixels", str(200 * 32 * 32)], check=True, env=env)
    assert json.load(open(tmp_path / "ev" / "exp_eval.json"))["n"] == 1


def test_cli(training_ckpt, tmp_path):
    ckpt, _, _, _ = training_ckpt
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(["src", "tests", "."]))
    subprocess.run([sys.executable, "-m", "attn_ft.export", ckpt, str(tmp_path / "cli")], check=True, env=env)
    assert os.path.exists(tmp_path / "cli" / "config.json")


@pytest.fixture
def tied_training_ckpt(tiny_dir, tmp_path):
    """Like training_ckpt, but with tied word embeddings (Qwen3.5-2B layout)."""
    from conftest import tiny_config
    tok = AutoTokenizer.from_pretrained(tiny_dir)
    tok.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    proc = AutoProcessor.from_pretrained(tiny_dir)
    proc.tokenizer = tok
    cfg = tiny_config(vocab_size=len(tok))
    cfg.tie_word_embeddings = True
    cfg.text_config.tie_word_embeddings = True
    torch.manual_seed(0)
    m = Qwen3_5ForConditionalGenerationWithPointer(cfg).float()
    assert m.get_input_embeddings().weight.data_ptr() == m.get_output_embeddings().weight.data_ptr()
    ids = tok.convert_tokens_to_ids(ADDITIONAL_SPECIAL_TOKENS)
    frozen_rows = m.get_input_embeddings().weight.detach()[ids].clone()
    emb = PointerRowEmbedding.install(m, ids)
    with torch.no_grad():
        emb.rows.normal_()
    out = tmp_path / "tied-checkpoint"
    m.save_pretrained(out)
    proc.save_pretrained(out)
    tok.save_pretrained(out)
    return str(out), m, ids, frozen_rows


def test_tied_export_reproduces_trained_model(tied_training_ckpt, tmp_path):
    ckpt, m, ids, frozen_rows = tied_training_ckpt
    out = export_bf16(ckpt, str(tmp_path / "tied_exp"), dtype=torch.float32)
    x = torch.tensor([[1, 2, ids[2], ids[4], ids[3], 5]])
    with torch.no_grad():
        want = m.eval()(input_ids=x).logits
    loaded, _ = load_pointer_model(out, device="cpu")
    with torch.no_grad():
        got = loaded(input_ids=x).logits
    assert torch.allclose(got, want, atol=1e-5)
    assert loaded.get_input_embeddings().weight.data_ptr() != loaded.get_output_embeddings().weight.data_ptr()
    assert torch.equal(loaded.get_output_embeddings().weight[ids], frozen_rows)          # lm_head kept frozen rows
    assert torch.equal(loaded.get_input_embeddings().weight[ids], m.get_input_embeddings().rows.detach())
    cfg = json.load(open(os.path.join(out, "config.json")))
    assert cfg.get("tie_word_embeddings") is False and cfg["text_config"].get("tie_word_embeddings") is False
