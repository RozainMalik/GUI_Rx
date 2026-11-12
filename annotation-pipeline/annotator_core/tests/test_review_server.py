import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from annotator.grading import load_verdicts
from tests.conftest import INSTR, make_data
from tools import review_server as rs


def test_review_server_roundtrip(tmp_path):
    root = make_data(tmp_path, 1)
    sample = tmp_path / "sample.json"
    items = [{"image": "IrfanView/IrfanView_001.png", "app": "IrfanView", "frame": "IrfanView_001.png", "frame_index": 1,
              "tier": "A", "bbox_px": [288, 162, 576, 216], "element": "Slideshow...", "element_kind": "menu item", "instructions": INSTR},
             {"image": "Missing/Missing_001.png", "app": "Missing", "frame": "Missing_001.png", "frame_index": 1,
              "tier": "A", "bbox_px": [1, 1, 5, 5], "element": "x", "element_kind": "tab", "instructions": INSTR}]
    sample.write_text(json.dumps({"items": items}), encoding="utf-8")
    assert len(rs.load_items(sample, root)) == 1 and len(rs.load_items(sample, root, only_local=False)) == 2

    verdicts = tmp_path / "verdicts.jsonl"
    server = rs.make_server(rs.load_items(sample, root), root, verdicts, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert "MedSPOT box review" in urlopen(base + "/").read().decode()
        data = json.loads(urlopen(base + "/api/items").read())
        assert data["reviewed"] == 0 and data["items"][0]["reviewed"] is False and data["items"][0]["verdict"] is None
        body = json.dumps({"image": "IrfanView/IrfanView_001.png", "verdict": "loose",
                           "bbox_px": [288, 162, 576, 216], "original_bbox_px": [288, 162, 576, 216]}).encode()
        resp = json.loads(urlopen(Request(base + "/api/verdict", data=body, headers={"Content-Type": "application/json"})).read())
        assert resp == {"ok": True, "reviewed": 1}
        assert json.loads(urlopen(base + "/api/items").read())["items"][0]["verdict"] == "loose"
        assert urlopen(base + "/shot/IrfanView/IrfanView_001.png").headers["Content-Type"] == "image/png"
        try:
            urlopen(base + "/shot/nope.png")
            assert False, "expected 404"
        except HTTPError as e:
            assert e.code == 404
        bad = json.dumps({"image": "IrfanView/IrfanView_001.png", "verdict": "meh"}).encode()
        try:
            urlopen(Request(base + "/api/verdict", data=bad, headers={"Content-Type": "application/json"}))
            assert False, "expected 400"
        except HTTPError as e:
            assert e.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    v = load_verdicts(verdicts)
    assert len(v) == 1 and list(v.values())[0]["verdict"] == "loose" and "at" in list(v.values())[0]
