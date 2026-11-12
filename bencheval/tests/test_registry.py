import json
import pytest
from benchev.registry import Row, load, append, DuplicateRow


def _row(**kw):
    base = dict(model_id="m", bench="ssv2", split="all", mode="1step", protocol_id="ssv2@v1",
                metrics={"acc": 91.4}, n=1272, provenance="imported", source="x.json", date="2026-10-06")
    base.update(kw)
    return Row(**base)


def test_roundtrip(tmp_path):
    p = tmp_path / "registry.jsonl"
    append(p, [_row(), _row(split="mobile")])
    rows = load(p)
    assert [r.split for r in rows] == ["all", "mobile"]
    assert rows[0].metrics == {"acc": 91.4}


def test_duplicate_refused(tmp_path):
    p = tmp_path / "registry.jsonl"
    append(p, [_row()])
    with pytest.raises(DuplicateRow):
        append(p, [_row(metrics={"acc": 1.0})])
    assert len(load(p)) == 1


def test_duplicate_within_one_batch_refused(tmp_path):
    with pytest.raises(DuplicateRow):
        append(tmp_path / "r.jsonl", [_row(), _row()])


def test_supersede_moves_old_row(tmp_path):
    p = tmp_path / "registry.jsonl"
    append(p, [_row()])
    append(p, [_row(metrics={"acc": 92.0})], supersede=True)
    assert [r.metrics["acc"] for r in load(p)] == [92.0]
    old = [json.loads(l) for l in (tmp_path / "registry_superseded.jsonl").read_text().splitlines()]
    assert old[0]["metrics"]["acc"] == 91.4


def test_same_key_different_provenance_coexist(tmp_path):
    p = tmp_path / "registry.jsonl"
    append(p, [_row(), _row(provenance="harness")])
    assert len(load(p)) == 2


def test_bad_provenance_rejected():
    with pytest.raises(ValueError):
        _row(provenance="guess")


def test_write_is_atomic_no_tmp_left(tmp_path):
    reg = tmp_path / "registry.jsonl"
    append(reg, [_row()])
    assert [x.name for x in tmp_path.iterdir()] == ["registry.jsonl"]


def test_supersede_keeps_verified_when_metrics_unchanged(tmp_path):
    reg = tmp_path / "registry.jsonl"
    old = _row()
    old.parity = "verified"
    append(reg, [old])
    same, changed = _row(), _row()
    same.parity = "unverified"
    append(reg, [same], supersede=True, keep_verified=True)
    assert load(reg)[0].parity == "verified"
    changed.parity, changed.metrics = "unverified", {"acc": 1.0}
    append(reg, [changed], supersede=True, keep_verified=True)
    assert load(reg)[0].parity == "unverified"
