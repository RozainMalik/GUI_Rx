import json

import pytest
from test_dataset_qwen35 import sample_json, tiny_dir  # noqa: F401
from test_train_attn import _run
from transformers import AutoTokenizer


def test_second_stage_starts_from_exported_first_stage(tiny_dir, sample_json, tmp_path):
    s1, s2 = tmp_path / "s1", tmp_path / "s2"
    _run(tiny_dir, sample_json, s1, ["--max_steps", "2", "--audit_params", "True"])
    init = s1 / "final_bf16"
    _run(str(init), sample_json, s2, ["--max_steps", "2", "--audit_params", "True"])
    a1, a2 = json.load(open(s1 / "param_audit.json")), json.load(open(s2 / "param_audit.json"))
    # every parameter starts where stage 1 ended; the frozen table differs only by the folded-in pointer rows
    for name, fp in a1["end"].items():
        if name.endswith("embed_tokens.weight"):
            continue
        assert a2["begin"][name] == pytest.approx(fp, rel=1e-6), name
    assert a2["unexpected"] == []
    assert any(".self_attn.q_proj." in n for n in a2["changed"])
    t1, t2 = (json.load(open(d / "run_summary.json"))["trainable_params"] for d in (s1, s2))
    assert t1 == t2
    assert len(AutoTokenizer.from_pretrained(str(init))) == len(AutoTokenizer.from_pretrained(str(s2 / "final_bf16")))
