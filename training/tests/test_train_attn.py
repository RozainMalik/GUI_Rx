import glob
import json
import os
import subprocess
import sys

import pytest
from test_dataset_qwen35 import sample_json, tiny_dir  # noqa: F401


def _run(tiny_dir, sample_json, out, extra, check=True):
    cmd = [sys.executable, "train_attn.py", "--model_type", "qwen35", "--model_name_or_path", tiny_dir,
           "--data_path", str(sample_json / "train.json"), "--image_folder", str(sample_json),
           "--output_dir", str(out), "--per_device_train_batch_size", "1", "--learning_rate", "1e-3",
           "--lr_scheduler_type", "cosine", "--warmup_steps", "1",
           "--max_pixels", str(400 * 32 * 32), "--image_patch_size", "16", "--weighting", "query_1",
           "--pointer_loss_weight", "1.0", "--lm_loss_weight", "1.0", "--gradient_checkpointing", "True",
           "--bf16", "False", "--frozen_dtype", "float32", "--expect_trainable_range", "none",
           "--report_to", "none", "--dataloader_num_workers", "0", "--use_cpu", "True", "--logging_steps", "1",
           "--save_strategy", "steps", "--save_steps", "2", "--seed", "0"] + extra
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(["src", "tests", "."]), OMP_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if check:
        assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-6000:])
    return r


def _lrs(out):
    st = json.load(open(glob.glob(str(out / "checkpoint-*" / "trainer_state.json"))[-1]))
    return {e["step"]: e["learning_rate"] for e in st["log_history"] if "learning_rate" in e}


def test_short_run_changes_only_allowed_params_and_exports(tiny_dir, sample_json, tmp_path):
    out = tmp_path / "run"
    _run(tiny_dir, sample_json, out, ["--max_steps", "2", "--audit_params", "True"])
    audit = json.load(open(out / "param_audit.json"))
    assert audit["unexpected"] == []
    assert any("embed_tokens.rows" in n for n in audit["changed"])
    assert any(".self_attn.q_proj." in n for n in audit["changed"])
    summary = json.load(open(out / "run_summary.json"))
    assert summary["global_step"] == 2 and summary["trainable_params"] > 0
    assert os.path.exists(out / "final_bf16" / "config.json")


def test_stop_then_resume_keeps_schedule_and_restores_weights(tiny_dir, sample_json, tmp_path):
    ref = tmp_path / "ref"
    _run(tiny_dir, sample_json, ref, ["--max_steps", "4"])
    out = tmp_path / "split"
    _run(tiny_dir, sample_json, out, ["--max_steps", "4", "--stop_at_step", "2"])
    assert os.path.isdir(out / "checkpoint-2") and not os.path.exists(out / "final_bf16")
    assert json.load(open(out / "run_summary.json"))["global_step"] == 2
    from attn_ft.export import read_pointer_rows
    _, rows2 = read_pointer_rows(str(out / "checkpoint-2"))
    _run(tiny_dir, sample_json, out, ["--max_steps", "4", "--audit_params", "True"])
    audit = json.load(open(out / "param_audit.json"))
    begin_rows = [v for k, v in audit["begin"].items() if k.endswith("embed_tokens.rows")][0]
    assert begin_rows[0] == pytest.approx(float(rows2.float().sum()), rel=1e-6, abs=1e-8)   # resumed, not base (abs: sum ~0)
    assert json.load(open(out / "run_summary.json"))["global_step"] == 4
    ref_lr, split_lr = _lrs(ref), _lrs(out)
    for step in (1, 2, 3, 4):
        assert split_lr[step] == pytest.approx(ref_lr[step]), step


def test_resubmitted_stage1_exits_without_training(tiny_dir, sample_json, tmp_path):
    out = tmp_path / "again"
    _run(tiny_dir, sample_json, out, ["--max_steps", "4", "--stop_at_step", "2"])
    r = _run(tiny_dir, sample_json, out, ["--max_steps", "4", "--stop_at_step", "2"])
    assert "already at step 2 >= stop_at_step 2" in r.stdout + r.stderr
    assert not os.path.isdir(out / "checkpoint-4")


def test_trainable_range_assert(tiny_dir, sample_json, tmp_path):
    r = _run(tiny_dir, sample_json, tmp_path / "bad", ["--max_steps", "1", "--expect_trainable_range", "1e9,2e9"],
             check=False)
    assert r.returncode != 0 and "trainable parameter count" in r.stderr


def test_partial_checkpoint_is_ignored(tmp_path):
    """A checkpoint dir without trainer_state.json (killed mid-save) must not count as done."""
    from attn_ft.callbacks import latest_checkpoint
    (tmp_path / "checkpoint-2").mkdir()
    (tmp_path / "checkpoint-2" / "trainer_state.json").write_text("{}")
    (tmp_path / "checkpoint-4").mkdir()
    assert latest_checkpoint(str(tmp_path)) == (2, str(tmp_path / "checkpoint-2"))
    assert latest_checkpoint(str(tmp_path / "missing")) == (0, None)


def test_fingerprint_is_chunked_and_exact():
    """The audit must never copy a whole parameter to fp32 at once: a 2B embedding (248k x 2048) is 1.9 GiB in fp32.
    Chunked sums must equal the full-precision reference."""
    import torch
    from attn_ft import callbacks
    assert callbacks.FINGERPRINT_CHUNK <= 2 ** 24                      # <= 64 MiB of fp32 per step
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(3 * callbacks.FINGERPRINT_CHUNK + 7, dtype=torch.bfloat16))
    m = torch.nn.Module()
    m.w = p
    fp = callbacks._fingerprint(m)["w"]
    ref = [float(p.detach().double().sum()), float(p.detach().double().abs().sum())]
    assert abs(fp[0] - ref[0]) <= 1e-3 * (1 + abs(ref[0])) and abs(fp[1] - ref[1]) <= 1e-6 * ref[1]
