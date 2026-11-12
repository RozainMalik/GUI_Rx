import json

from annotator import cli
from annotator.calibrate import MARKER
from annotator.records import load_record, record_path
from tests.conftest import FakeGenerator, happy_router, make_data, make_frame
from tests.test_calibrate import make_dataset, router as calibrate_router


def test_annotate_refuses_without_calibration_marker(tmp_path, capsys):
    root = make_data(tmp_path, 1)
    rc = cli.main(["annotate", "--data-root", str(root), "--work", str(tmp_path / "work"), "--app", "IrfanView"])
    assert rc == 2 and MARKER in capsys.readouterr().out


def test_calibrate_command_writes_marker_and_reports(tmp_path, monkeypatch, capsys):
    root = make_dataset(tmp_path, 10)
    probe = tmp_path / "probe.png"
    make_frame().save(probe)
    monkeypatch.setattr(cli, "_generator", lambda args: FakeGenerator(router=calibrate_router))
    rc = cli.main(["calibrate", "--dataset", str(root), "--probe", str(probe), "--work", str(tmp_path / "work"), "--n", "10"])
    out = capsys.readouterr().out
    assert rc == 0 and "CALIBRATION_OK" in out and "hits=8/10" in out and (tmp_path / "work" / MARKER).exists()
    monkeypatch.setattr(cli, "_generator", lambda args: FakeGenerator(router=calibrate_router, prompt_tokens=900))
    rc = cli.main(["calibrate", "--dataset", str(root), "--probe", str(probe), "--work", str(tmp_path / "w2"), "--n", "10"])
    assert rc == 2 and "RESOLUTION_FAIL" in capsys.readouterr().out and not (tmp_path / "w2" / MARKER).exists()


def test_annotate_all_export_qc_apply_grades(tmp_path, monkeypatch, capsys):
    root = make_data(tmp_path, 2)
    work = tmp_path / "work"
    monkeypatch.setattr(cli, "_generator", lambda args: FakeGenerator(router=happy_router()))
    rc = cli.main(["annotate", "--data-root", str(root), "--work", str(work), "--all", "--skip-calibration-check", "--run-id", "t1"])
    out = capsys.readouterr().out
    assert rc == 0 and "RECORDS_WRITTEN IrfanView frames=2" in out and "ANNOTATE_DONE" in out and "shipped=2" in out
    assert load_record(record_path(work, "IrfanView", 1))["provenance"]["run_id"] == "t1"

    rc = cli.main(["export", "--work", str(work), "--out", str(tmp_path / "exp"), "--image-root", "/r"])
    assert rc == 0 and "EXPORTED" in capsys.readouterr().out
    assert len(json.loads((tmp_path / "exp" / "medspot_train.json").read_text())["tasks"]) == 6
    rc = cli.main(["export", "--work", str(work), "--out", str(tmp_path / "exp_flat"), "--image-root", "/r", "--flat", "--tier", "C"])
    assert rc == 0 and (tmp_path / "exp_flat" / "medspot_train_flat.json").exists()

    rc = cli.main(["qc", "--work", str(work), "--data-root", str(root), "--out", str(tmp_path / "reports"), "--n", "1"])
    assert rc == 0 and (tmp_path / "reports" / "qc_stats.json").exists()

    rc = cli.main(["apply-grades", "--work", str(work)])
    assert rc == 0 and "APPLIED_GRADES 0" in capsys.readouterr().out


def test_reset_stage_and_parse_only(tmp_path, monkeypatch, capsys):
    root = make_data(tmp_path, 1)
    work = tmp_path / "work"
    monkeypatch.setattr(cli, "_generator", lambda args: FakeGenerator(router=happy_router()))
    cli.main(["annotate", "--data-root", str(root), "--work", str(work), "--app", "IrfanView", "--skip-calibration-check"])
    rc = cli.main(["reset", "--work", str(work), "--app", "IrfanView", "--stage", "verify"])
    assert rc == 0 and "RESET IrfanView records=1 stage=verify" in capsys.readouterr().out
    rec = load_record(record_path(work, "IrfanView", 1))
    assert rec["stages_done"] == ["propose", "describe", "ground"] and rec["candidates"][0]["verify"] is None
    rc = cli.main(["annotate", "--data-root", str(root), "--work", str(work), "--app", "IrfanView", "--skip-calibration-check"])
    assert rc == 0 and load_record(record_path(work, "IrfanView", 1))["status"] == "shipped"
    rc = cli.main(["reset", "--work", str(work), "--app", "IrfanView", "--stage", "ground", "--parse-only"])
    assert rc == 0 and "changed=0" in capsys.readouterr().out
    rc = cli.main(["reset", "--work", str(work), "--app", "IrfanView", "--stage", "refine", "--parse-only"])
    assert rc == 2


def test_annotate_transport_errors_exit_nonzero(tmp_path, monkeypatch, capsys):
    root = make_data(tmp_path, 1)
    from annotator.calling import TransportError

    def dead(call):
        raise TransportError("gone")
    monkeypatch.setattr(cli, "_generator", lambda args: FakeGenerator(router=dead))
    rc = cli.main(["annotate", "--data-root", str(root), "--work", str(tmp_path / "work"), "--apps", "IrfanView", "--skip-calibration-check"])
    out = capsys.readouterr().out
    assert rc == 1 and "ANNOTATE_INCOMPLETE" in out and "ANNOTATE_DONE" not in out


def test_apply_grades_with_no_verdicts_is_a_noop(tmp_path, monkeypatch, capsys):
    root = make_data(tmp_path, 2)
    work = tmp_path / "work"
    monkeypatch.setattr(cli, "_generator", lambda args: FakeGenerator(router=happy_router()))
    rc = cli.main(["annotate", "--data-root", str(root), "--work", str(work), "--app", "IrfanView", "--skip-calibration-check"])
    assert rc == 0
    capsys.readouterr()
    assert not (work / "review").exists()             # fresh work dir: no verdict log, no sample file
    rc = cli.main(["apply-grades", "--work", str(work)])
    out = capsys.readouterr().out
    assert rc == 0 and "APPLIED_GRADES 0" in out and "Traceback" not in out


def test_grade_builds_the_sample_without_serving(tmp_path, monkeypatch, capsys):
    root = make_data(tmp_path, 2)
    work = tmp_path / "work"
    monkeypatch.setattr(cli, "_generator", lambda args: FakeGenerator(router=happy_router()))
    cli.main(["annotate", "--data-root", str(root), "--work", str(work), "--app", "IrfanView", "--skip-calibration-check"])
    monkeypatch.setattr(cli, "_serve", lambda *a, **k: 0)
    rc = cli.main(["grade", "--work", str(work), "--root", str(root), "--n", "1", "--no-open"])
    assert rc == 0 and (work / "review" / "sample_AB.json").exists() and "sample of 1 rows" in capsys.readouterr().out


def test_annotate_batch_takes_one_seeded_frame_per_app_and_reports(tmp_path, monkeypatch, capsys):
    from annotator.frames import batch_frame
    root = make_data(tmp_path, 4)
    make_data(tmp_path, 3, app="Mango")
    work = tmp_path / "work"
    monkeypatch.setattr(cli, "_generator", lambda args: FakeGenerator(router=happy_router()))
    rc = cli.main(["annotate", "--data-root", str(root), "--work", str(work), "--all", "--skip-calibration-check",
                   "--batch", "2", "--batch-seed", "7", "--run-id", "qwen_batch2"])
    out = capsys.readouterr().out
    assert rc == 0 and "BATCH 2 seed=7 apps=2 frames=2" in out and "ANNOTATE_DONE" in out
    for app in ("IrfanView", "Mango"):
        want = batch_frame(root, app, 2, 7).index
        assert [p.stem for p in (work / "records" / app).glob("*.json")] == [f"{want:03d}"]

    rc = cli.main(["batch-report", "--work", str(work), "--data-root", str(root), "--batch", "2", "--batch-seed", "7",
                   "--out", str(tmp_path / "qwen_batches")])
    assert rc == 0 and "BATCH_REPORT batch=2 frames=2 shipped=2" in capsys.readouterr().out
    assert len(list((tmp_path / "qwen_batches" / "batch_002").glob("*.png"))) == 2


def test_annotate_batch_skips_an_exhausted_app_and_refuses_limit(tmp_path, monkeypatch, capsys):
    root = make_data(tmp_path, 1)
    make_data(tmp_path, 3, app="Mango")
    monkeypatch.setattr(cli, "_generator", lambda args: FakeGenerator(router=happy_router()))
    base = ["annotate", "--data-root", str(root), "--work", str(tmp_path / "work"), "--all", "--skip-calibration-check"]
    rc = cli.main(base + ["--batch", "2"])
    out = capsys.readouterr().out
    assert rc == 0 and "BATCH_SKIP IrfanView frames=1" in out and "BATCH 2 seed=20260923 apps=1 frames=1" in out
    assert cli.main(base + ["--batch", "1", "--limit", "3"]) == 2
    assert "--batch" in capsys.readouterr().out
