import hashlib

from benchev import ROOT

V = ROOT / "vendor" / "upstream_gui_actor"


def test_vendored_files_match_manifest():
    lines = (V / "SHA256SUMS").read_text(encoding="utf-8").split("\n")
    entries = [ln.split(None, 1) for ln in lines if ln.strip()]
    assert len(entries) == 9                      # 8 package files + LICENSE
    for digest, rel in entries:
        assert hashlib.sha256((V / rel.strip()).read_bytes()).hexdigest() == digest, rel


def test_vendor_md_records_commit():
    assert "d98d1bbd01862f9112114b83b032f492c365a173" in (ROOT / "vendor" / "VENDOR.md").read_text(encoding="utf-8")
