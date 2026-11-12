import io
import json
import subprocess
import sys

from PIL import Image

from scripts.audit_mix import boxes, record_problem


def _rec(image, *bbs):
    convs = [{"from": "human", "value": "<image> x"}]
    convs += [{"from": "gpt", "value": "c", "recipient": "os", "end_turn": True, "bbox_gt": b} for b in bbs]
    return {"image": image, "conversations": convs}


def test_boxes_single_multi_and_none():
    assert boxes(_rec("a.png", [0.1, 0.1, 0.2, 0.2])) == [[0.1, 0.1, 0.2, 0.2]]
    assert boxes(_rec("a.png", [[0, 0, 1, 1], [0.2, 0.2, 0.3, 0.3]], None)) == [[0, 0, 1, 1], [0.2, 0.2, 0.3, 0.3]]


def test_record_problem():
    assert record_problem(_rec("a.png", [0.1, 0.1, 0.2, 0.2])) is None
    assert record_problem(_rec("a.png", None)) == "no bbox_gt"
    assert record_problem(_rec("a.png", [0.3, 0.1, 0.2, 0.2])) == "inverted box"
    assert record_problem(_rec("a.png", [0.1, 0.1, 0.2])) == "malformed box"


def _png(color):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, "PNG")
    return buf.getvalue()


A, B, Z = _png("red"), _png("blue"), _png("green")


def _mix(tmp_path, recs, bench_bytes):
    img = tmp_path / "img"
    img.mkdir(exist_ok=True)
    for name, data in {"a.png": A, "b.png": B}.items():
        (img / name).write_bytes(data)
    (tmp_path / "s.json").write_text(json.dumps(recs))
    (tmp_path / "mix.yaml").write_text(f"datasets:\n  - json_path: {tmp_path / 's.json'}\n"
                                       f"    images_folder: {img}/\n    sampling_strategy: \"all\"\n")
    bench = tmp_path / "bench"
    (bench / "sub").mkdir(parents=True, exist_ok=True)
    (bench / "sub" / "x.png").write_bytes(bench_bytes)
    return [sys.executable, "scripts/audit_mix.py", "--data", str(tmp_path / "mix.yaml"), "--bench", str(bench)]


def test_clean_mix_exits_zero(tmp_path):
    cmd = _mix(tmp_path, [_rec("a.png", [0.1, 0.1, 0.2, 0.2])], Z)
    assert subprocess.run(cmd + ["--expect_records", "1"], capture_output=True, text=True).returncode == 0


def test_missing_image_and_wrong_count_fail(tmp_path):
    cmd = _mix(tmp_path, [_rec("gone.png", [0.1, 0.1, 0.2, 0.2])], Z)
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 1 and "missing images: 1" in r.stdout
    cmd = _mix(tmp_path, [_rec("a.png", [0.1, 0.1, 0.2, 0.2])], Z)
    assert subprocess.run(cmd + ["--expect_records", "2"], capture_output=True, text=True).returncode == 1


def test_leak_fails_then_drop_leaks_rewrites_json(tmp_path):
    recs = [_rec("a.png", [0.1, 0.1, 0.2, 0.2]), _rec("b.png", [0.1, 0.1, 0.2, 0.2])]
    cmd = _mix(tmp_path, recs, B)                                   # b.png is byte-identical to a benchmark image
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 1 and "leaked records: 1" in r.stdout
    assert subprocess.run(cmd + ["--drop_leaks"], capture_output=True, text=True).returncode == 0
    assert [x["image"] for x in json.loads((tmp_path / "s.json").read_text())] == ["a.png"]
    assert len(json.loads((tmp_path / "s.orig.json").read_text())) == 2
    assert subprocess.run(cmd, capture_output=True, text=True).returncode == 0


def test_empty_or_truncated_image_fails(tmp_path):
    """Zero-byte images (an interrupted write) must fail the audit, not just be hashed."""
    recs = [_rec("a.png", [0.1, 0.1, 0.2, 0.2]), _rec("b.png", [0.1, 0.1, 0.2, 0.2])]
    cmd = _mix(tmp_path, recs, Z)
    (tmp_path / "img" / "a.png").write_bytes(b"")
    (tmp_path / "img" / "b.png").write_bytes(B[:len(B) // 2])
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 1 and "corrupt images: 2" in r.stdout


def test_empty_or_missing_bench_dir_fails(tmp_path):
    """A --bench dir with no images (typo, unmounted path) must fail instead of leaving the leak check vacuous."""
    cmd = _mix(tmp_path, [_rec("a.png", [0.1, 0.1, 0.2, 0.2])], Z)
    (tmp_path / "empty").mkdir()
    for extra in (["--bench", str(tmp_path / "empty")], ["--bench", str(tmp_path / "nope")]):
        r = subprocess.run(cmd + extra, capture_output=True, text=True)
        assert r.returncode == 1 and "no benchmark images" in r.stdout
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0 and f"{tmp_path / 'bench'}: 1 images" in r.stdout


def test_same_json_listed_twice_is_audited_once(tmp_path):
    """--drop_leaks with the same JSON in two yamls must not apply the leak indices twice (it would drop good records)."""
    recs = [_rec("b.png", [0.1, 0.1, 0.2, 0.2]), _rec("a.png", [0.1, 0.1, 0.2, 0.2])]
    cmd = _mix(tmp_path, recs, B)
    r = subprocess.run(cmd + ["--data", str(tmp_path / "mix.yaml"), "--drop_leaks"], capture_output=True, text=True)
    assert r.returncode == 0 and "records: 1\n" in r.stdout
    assert [x["image"] for x in json.loads((tmp_path / "s.json").read_text())] == ["a.png"]
