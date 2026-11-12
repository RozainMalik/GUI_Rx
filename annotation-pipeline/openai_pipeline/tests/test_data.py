from pathlib import Path

from oai import data as d
from oai.config import load_config
from tests.conftest import make_data, make_frame, write_config


def test_missing_apps_are_the_thirteen_from_the_spec():
    assert len(d.MISSING_APPS) == 13 and "3D_Slicer" in d.MISSING_APPS and "Weasis" in d.MISSING_APPS


def test_pull_tree_builds_the_ssh_tar_pipe_and_is_idempotent(tmp_path):
    calls = []

    def fake_pipe(remote_cmd, local_cmd):
        calls.append((remote_cmd, local_cmd))
        make_data(Path(local_cmd[-1]), apps=("QuPath",), n=1)
        return 0
    assert d.pull_tree("u@h", "remote/data", "QuPath", tmp_path / "root", runner=fake_pipe) == 0
    remote_cmd, local_cmd = calls[0]
    assert remote_cmd[:3] == ["ssh", "-o", "BatchMode=yes"] and remote_cmd[3] == "u@h"
    assert remote_cmd[4] == "tar czf - -C remote/data QuPath"
    assert local_cmd[:4] == ["tar", "xzf", "-", "-C"] and local_cmd[4] == str(tmp_path / "root" / ".QuPath.partial")
    assert (tmp_path / "root" / "QuPath").is_dir() and not (tmp_path / "root" / ".QuPath.partial").exists()
    assert d.pull_tree("u@h", "remote/data", "QuPath", tmp_path / "root", runner=fake_pipe) == 0 and len(calls) == 1   # already there


def test_pull_tree_discards_a_partial_pull_and_retries(tmp_path):
    calls = []

    def fake_pipe(remote_cmd, local_cmd):
        calls.append((remote_cmd, local_cmd))
        (Path(local_cmd[-1]) / "QuPath").mkdir(parents=True)
        make_frame().save(Path(local_cmd[-1]) / "QuPath" / "QuPath_001.png")
        return 1
    assert d.pull_tree("u@h", "remote/data", "QuPath", tmp_path / "root", runner=fake_pipe) == 1
    assert not (tmp_path / "root" / "QuPath").exists() and not (tmp_path / "root" / ".QuPath.partial").exists()
    assert d.pull_tree("u@h", "remote/data", "QuPath", tmp_path / "root", runner=fake_pipe) == 1 and len(calls) == 2


def test_ensure_junction_only_when_missing(tmp_path):
    calls = []
    target = tmp_path / "images"
    target.mkdir()
    link = tmp_path / "data"

    def fake_cmd(cmd):
        calls.append(cmd)
        link.mkdir()
        return 0
    assert d.ensure_junction(link, target, runner=fake_cmd) is True
    assert calls == [["cmd", "/c", "mklink", "/J", str(link), str(target)]]
    assert d.ensure_junction(link, target, runner=fake_cmd) is False and len(calls) == 1


def test_init_data_pulls_what_is_missing_and_prints_data_ok(tmp_path, capsys):
    cfg = load_config(write_config(tmp_path, batches="expected_apps = 3"))
    make_data(cfg.image_root, apps=("Present",), n=2)
    pulled = []

    def fake_pipe(remote_cmd, local_cmd):
        name = remote_cmd[4].split()[-1]                       # the folder name is the last token
        pulled.append(name)
        part = Path(local_cmd[-1])
        if name == "MedicalDataset":
            root = part / "MedicalDataset"
            (root / "Annotations").mkdir(parents=True)
            (root / "Images" / "X").mkdir(parents=True)
            make_frame().save(root / "Images" / "X" / "Screenshot_1.png")
            (root / "Annotations" / "X_Annotation.json").write_text("{}", encoding="utf-8")
        else:
            make_data(part, apps=(name,), n=1)
        return 0
    counts = d.init_data(cfg, pipe=fake_pipe, cmd=lambda cmd: 0, apps=("Present", "3D_Slicer", "Weasis"))
    assert pulled == ["3D_Slicer", "Weasis", "MedicalDataset"]
    assert counts == {"apps": 3, "frames": 4, "benchmark_images": 1}
    assert "DATA_OK apps=3 frames=4 benchmark_images=1" in capsys.readouterr().out
    assert d.init_data(cfg, pipe=fake_pipe, cmd=lambda cmd: 0, apps=("Present", "3D_Slicer", "Weasis"))["apps"] == 3 and len(pulled) == 3


def test_init_data_reports_incomplete_when_a_pull_fails(tmp_path, capsys):
    cfg = load_config(write_config(tmp_path))
    out = d.init_data(cfg, pipe=lambda r, l: 1, cmd=lambda c: 0, apps=("Weasis",))
    assert out["apps"] == 0 and "DATA_INCOMPLETE" in capsys.readouterr().out


def test_init_data_reports_a_failed_junction(tmp_path, capsys):
    data, images = (str(tmp_path / p).replace("\\", "/") for p in ("data", "images"))
    paths = f'data = "{data}"\nbenchmark = "benchmark"\nwork = "work"\nexport = "export"\nreports = "reports"\nimage_root = "{images}"'
    cfg = load_config(write_config(tmp_path, paths=paths))
    make_data(cfg.image_root, apps=("Present",), n=1)
    (cfg.benchmark / "Annotations").mkdir(parents=True)
    (cfg.benchmark / "Annotations" / "X_Annotation.json").write_text("{}", encoding="utf-8")
    out = d.init_data(cfg, pipe=lambda r, l: 0, cmd=lambda c: 1, apps=("Present",))
    assert out["apps"] == 1 and "DATA_INCOMPLETE apps=1 frames=1 benchmark_images=0 failed=junction" in capsys.readouterr().out
