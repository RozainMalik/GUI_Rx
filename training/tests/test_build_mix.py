import io
import json
import tarfile

import pytest

from scripts.build_mix import SOURCES, ConcatStream, extract_tar_stream, load_rows, select, zip_plan


def test_sources_match_the_spec():
    assert list(SOURCES) == ["guiact", "androidcontrol", "wave_ui", "uground", "gta"]
    assert sum(s["n"] for s in SOURCES.values()) == 254779
    assert SOURCES["uground"]["take"] == "first:60000"
    assert all(s["take"] == "all" for k, s in SOURCES.items() if k != "uground")
    assert all(("zip" in s) != ("tar" in s) for s in SOURCES.values())
    assert SOURCES["gta"]["inner"] == "gta_data/image"


def test_select():
    assert select(list(range(10)), "all") == list(range(10))
    assert select(list(range(10)), "first:3") == [0, 1, 2]
    with pytest.raises(ValueError):
        select(list(range(10)), "random:3")


def test_load_rows_local_applies_take(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps([{"i": i} for i in range(5)]))
    assert load_rows(str(p), "first:2") == [{"i": 0}, {"i": 1}]
    assert len(load_rows(str(p), "all")) == 5


def test_zip_plan_counts_compressed_bytes_and_missing():
    members = {"in/a.png": (0, 10, 10, 0, 0), "in/sub/b.png": (8, 5, 9, 0, 0), "other/c.png": (0, 7, 7, 0, 0)}
    assert zip_plan(members, "in", ["a.png", "sub/b.png", "c.png"]) == (["c.png"], 15)


def _tar_bytes(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in files.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def test_extract_tar_stream_over_split_parts(tmp_path):
    blob = _tar_bytes({"gta_data/image/x/1.png": b"one" * 400, "gta_data/image/2.png": b"two",
                       "gta_data/other/3.png": b"three"})
    cuts = [0, 700, 1500, len(blob)]                       # boundaries fall inside headers and file bodies
    parts = []
    for i in range(3):
        p = tmp_path / f"part_a{i}"
        p.write_bytes(blob[cuts[i]:cuts[i + 1]])
        parts.append(p.as_uri())
    out = tmp_path / "img"
    got = extract_tar_stream(io.BufferedReader(ConcatStream(parts), 1 << 20), "gta_data/image", {"x/1.png"}, str(out))
    assert got == {"x/1.png"}
    assert (out / "x" / "1.png").read_bytes() == b"one" * 400
    assert not (out / "2.png").exists()
    assert not list(out.rglob("*.part"))


def test_tar_urls_local_parts_dir(tmp_path):
    """--gta_parts_dir: the parts downloaded in parallel beforehand are read locally, in part order."""
    from scripts.build_mix import tar_urls
    s = SOURCES["gta"]
    assert tar_urls(s, None) == s["tar"]
    for u in reversed(s["tar"]):
        (tmp_path / u.rsplit("/", 1)[1]).write_bytes(b"x")
    assert tar_urls(s, str(tmp_path)) == [(tmp_path / u.rsplit("/", 1)[1]).as_uri() for u in s["tar"]]
    (tmp_path / "gta_data_part_ad").unlink()
    with pytest.raises(SystemExit):
        tar_urls(s, str(tmp_path))
