import json
import shutil
from pathlib import Path

from PIL import Image

from benchev.adapters.fake import FakeAdapter
from benchev.ingest import rows_from_run
from benchev.run import run


def _ssv2(root, n=4):
    rows = []
    for i in range(n):
        Image.new("RGB", (100, 100)).save(root / f"{i}.png")
        rows.append({"img_filename": f"{i}.png", "bbox": [0, 0, 50, 50], "instruction": "x", "split": "web",
                     "data_type": "icon"})
    json.dump(rows, open(root / "subset.json", "w"))


def _data(tmp_path):
    d = tmp_path / "data" / "screenspot_v2_full"
    d.mkdir(parents=True)
    _ssv2(d)
    return tmp_path / "data"


def test_full_run_scores_and_writes(tmp_path):
    data = _data(tmp_path)
    s = run("ours-2b-ab-clin", "ssv2", "1step", data, tmp_path / "raw", adapter=FakeAdapter(lambda i, t: (.2, .2)))
    assert s["metrics"]["acc"] == 100.0 and s["n_scored"] == 4
    # n_expected is the protocol's 1272 -> a 4-sample synthetic run is partial
    assert s["status"] == "partial"
    run_json = json.loads((tmp_path / "raw/ours-2b-ab-clin/ssv2/1step/run.json").read_text())
    assert run_json["protocol_id"] == "ssv2@v1" and "harness_commit" in run_json


def test_resume_torn_line_and_foreign_ids(tmp_path):
    data = _data(tmp_path)
    out = tmp_path / "raw/ours-2b-ab-clin/ssv2/1step"
    out.mkdir(parents=True)
    (out / "predictions.jsonl").write_text(
        json.dumps({"sid": "0000_0", "point": [.2, .2], "image_size": [100, 100]}) + "\n" +
        json.dumps({"sid": "FOREIGN", "point": [.9, .9], "image_size": [100, 100]}) + "\n" + '{"sid": "0001_1", "po')
    a = FakeAdapter(lambda i, t: (.2, .2))
    s = run("ours-2b-ab-clin", "ssv2", "1step", data, tmp_path / "raw", adapter=a)
    assert len(a.calls) == 3            # 0000_0 skipped; torn 0001_1 re-run; 0002, 0003
    assert s["n_scored"] == 4           # FOREIGN ignored


def test_ingest_partial_rows_flagged(tmp_path):
    data = _data(tmp_path)
    run("ours-2b-ab-clin", "ssv2", "1step", data, tmp_path / "raw", adapter=FakeAdapter(lambda i, t: (.2, .2)))
    rows = rows_from_run(tmp_path / "raw/ours-2b-ab-clin/ssv2/1step")
    assert {r.status for r in rows} == {"partial"} and {r.provenance for r in rows} == {"harness"}
    assert {r.split for r in rows} >= {"all", "group=web"}




def test_commit_fallback(tmp_path, monkeypatch):
    import benchev.run as r
    monkeypatch.setattr(r, "ROOT", tmp_path)          # not a git repo -> git prints nothing
    (tmp_path / ".harness_commit").write_text("abc123\n")
    assert r._commit() == "abc123"


def test_limit_run_gets_its_own_dir(tmp_path):
    data = _data(tmp_path)
    f = FakeAdapter(lambda i, t: (.2, .2))
    run("ours-2b-ab-clin", "ssv2", "1step", data, tmp_path / "raw", adapter=f)
    s = run("ours-2b-ab-clin", "ssv2", "1step", data, tmp_path / "raw", adapter=f, limit=2)
    assert s["n_scored"] == 2 and (tmp_path / "raw/ours-2b-ab-clin/ssv2/1step_limit2/summary.json").exists()
    full = json.loads((tmp_path / "raw/ours-2b-ab-clin/ssv2/1step/summary.json").read_text())
    assert full["n_scored"] == 4            # the full run's summary is untouched


def test_resume_refuses_other_checkpoint(tmp_path):
    import pytest
    data = _data(tmp_path)
    run("ours-2b-ab-clin", "ssv2", "1step", data, tmp_path / "raw", adapter=FakeAdapter(lambda i, t: (.2, .2)))
    rj = tmp_path / "raw/ours-2b-ab-clin/ssv2/1step/run.json"
    d = json.loads(rj.read_text())
    d["model"]["ckpt"] = "/somewhere/else"
    rj.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="ckpt"):
        run("ours-2b-ab-clin", "ssv2", "1step", data, tmp_path / "raw", adapter=FakeAdapter(lambda i, t: (.2, .2)))


def test_commit_marks_dirty_tree(monkeypatch):
    import benchev.run as r
    calls = iter([type("P", (), {"returncode": 0, "stdout": "abc\n"})(),
                  type("P", (), {"returncode": 1, "stdout": ""})()])     # git diff --quiet HEAD -> 1 = dirty
    monkeypatch.setattr(r.subprocess, "run", lambda *a, **k: next(calls))
    assert r._commit() == "abc-dirty"


def test_rescore_keeps_original_run_json(tmp_path, monkeypatch):
    """A re-score with no inference left must not overwrite the job id / runtime of the run that made the predictions."""
    data = _data(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "111")
    run("ours-2b-ab-clin", "ssv2", "1step", data, tmp_path / "raw", adapter=FakeAdapter(lambda i, t: (.2, .2)))
    rj = tmp_path / "raw/ours-2b-ab-clin/ssv2/1step/run.json"
    first = json.loads(rj.read_text())
    monkeypatch.setenv("SLURM_JOB_ID", "222")
    run("ours-2b-ab-clin", "ssv2", "1step", data, tmp_path / "raw", adapter=FakeAdapter(lambda i, t: (.9, .9)))
    after = json.loads(rj.read_text())
    assert after["slurm_job"] == "111" and after["seconds"] == first["seconds"] and after["finished"] == first["finished"]
    assert after["rescored"][-1]["slurm_job"] == "222" and "harness_commit" in after["rescored"][-1]


def _published(monkeypatch, **kw):
    import benchev.run as R
    from benchev.config import ModelCfg
    m = ModelCfg(id="ours-2b-ab-clin", kind="published", adapter="hf_chat", ckpt="/w", base=None,
                 status="published", **kw)
    monkeypatch.setattr(R, "load_model", lambda mid: m)
    return m


def test_run_passes_model_max_pixels(tmp_path, monkeypatch):
    _published(monkeypatch, max_pixels=1_003_520)
    a = FakeAdapter(lambda i, t: (.2, .2))
    run("ours-2b-ab-clin", "ssv2", "1step", _data(tmp_path), tmp_path / "raw", adapter=a)
    assert {c["max_pixels"] for c in a.calls} == {1_003_520}


def test_resume_refuses_changed_max_pixels(tmp_path, monkeypatch):
    import pytest
    data = _data(tmp_path)
    _published(monkeypatch, max_pixels=1_000_000)
    run("ours-2b-ab-clin", "ssv2", "1step", data, tmp_path / "raw", adapter=FakeAdapter(lambda i, t: (.2, .2)))
    (tmp_path / "raw/ours-2b-ab-clin/ssv2/1step/predictions.jsonl").write_text("")
    _published(monkeypatch, max_pixels=2_000_000)
    with pytest.raises(SystemExit, match="max_pixels"):
        run("ours-2b-ab-clin", "ssv2", "1step", data, tmp_path / "raw", adapter=FakeAdapter(lambda i, t: (.2, .2)))
    _published(monkeypatch, max_pixels=1_000_000, recipe="other")
    with pytest.raises(SystemExit, match="recipe"):
        run("ours-2b-ab-clin", "ssv2", "1step", data, tmp_path / "raw", adapter=FakeAdapter(lambda i, t: (.2, .2)))


def test_resume_accepts_old_run_json_without_new_keys(tmp_path):
    data = _data(tmp_path)
    run("ours-2b-ab-clin", "ssv2", "1step", data, tmp_path / "raw", adapter=FakeAdapter(lambda i, t: (.2, .2)))
    rj = tmp_path / "raw/ours-2b-ab-clin/ssv2/1step/run.json"
    d = json.loads(rj.read_text())
    for k in ("recipe", "max_pixels", "env", "flags", "parity"):
        d["model"].pop(k, None)
    rj.write_text(json.dumps(d))
    run("ours-2b-ab-clin", "ssv2", "1step", data, tmp_path / "raw", adapter=FakeAdapter(lambda i, t: (.2, .2)))


def test_ingest_notes_carry_override_and_flags(tmp_path, monkeypatch):
    _published(monkeypatch, max_pixels=1_003_520, flags=["no-published-reference"])
    monkeypatch.setenv("SLURM_JOB_ID", "9")
    run("ours-2b-ab-clin", "ssv2", "1step", _data(tmp_path), tmp_path / "raw", adapter=FakeAdapter(lambda i, t: (.2, .2)))
    rows = rows_from_run(tmp_path / "raw/ours-2b-ab-clin/ssv2/1step")
    assert rows[0].notes == "slurm 9; max_pixels=1003520 (model); no-published-reference"


def test_resume_refuses_changed_settings_after_a_crash(tmp_path, monkeypatch):
    # a run that dies mid-loop must still guard its predictions
    import pytest
    data = _data(tmp_path)
    _published(monkeypatch, max_pixels=1_000_000)
    calls = []

    def boom(img, ins):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("CUDA out of memory")
        return (.2, .2)
    with pytest.raises(RuntimeError):
        run("ours-2b-ab-clin", "ssv2", "1step", data, tmp_path / "raw", adapter=FakeAdapter(boom))
    _published(monkeypatch, max_pixels=2_000_000)
    with pytest.raises(SystemExit, match="max_pixels"):
        run("ours-2b-ab-clin", "ssv2", "1step", data, tmp_path / "raw", adapter=FakeAdapter(lambda i, t: (.2, .2)))


def test_resume_refuses_edited_recipe_prompt(tmp_path, monkeypatch):
    import dataclasses
    import pytest
    from benchev.adapters import recipes
    data = _data(tmp_path)
    _published(monkeypatch, recipe="uground")
    run("ours-2b-ab-clin", "ssv2", "1step", data, tmp_path / "raw", adapter=FakeAdapter(lambda i, t: (.2, .2)))
    (tmp_path / "raw/ours-2b-ab-clin/ssv2/1step/predictions.jsonl").write_text("")
    monkeypatch.setitem(recipes.RECIPES, "uground", dataclasses.replace(recipes.RECIPES["uground"], user="new {instruction}"))
    with pytest.raises(SystemExit, match="recipe"):
        run("ours-2b-ab-clin", "ssv2", "1step", data, tmp_path / "raw", adapter=FakeAdapter(lambda i, t: (.2, .2)))


def test_largest_smoke_takes_the_biggest_images(tmp_path):
    # the smoke must hit the largest frames to be an OOM check
    d = tmp_path / "data" / "screenspot_v2_full"
    d.mkdir(parents=True)
    rows = []
    for i, side in enumerate([100, 400, 50, 300, 200]):
        Image.new("RGB", (side, side)).save(d / f"{i}.png")
        rows.append({"img_filename": f"{i}.png", "bbox": [0, 0, 5, 5], "instruction": "x", "split": "web",
                     "data_type": "icon"})
    json.dump(rows, open(d / "subset.json", "w"))
    a = FakeAdapter(lambda i, t: (.2, .2))
    run("ours-2b-ab-clin", "ssv2", "1step", tmp_path / "data", tmp_path / "raw", adapter=a, limit=2, largest=True)
    assert sorted(c["size"] for c in a.calls) == [(300, 300), (400, 400)]
    assert (tmp_path / "raw/ours-2b-ab-clin/ssv2/1step_largest2/predictions.jsonl").exists()

