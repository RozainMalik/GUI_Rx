"""The data builders read ~150k remote ranges/files; one truncated HTTP response (IncompleteRead)
must be retried, not kill the build."""
import http.server
import io
import tarfile
import threading

import pytest

import scripts.build_groundcua as bg
import scripts.build_mix as bm
import scripts.remote_zip as rz


class _Server:
    """Serves one blob at /f (Range supported). The first `flaky` body responses are cut short after `cut` bytes."""

    def __init__(self, blob, flaky=1, cut=10):
        self.blob, self.flaky, self.cut, self.bodies = blob, flaky, cut, 0
        srv = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                start, end = 0, len(srv.blob) - 1
                rng = self.headers.get("Range")
                if rng:
                    a, b = rng.split("=")[1].split("-")
                    start, end = int(a), int(b) if b else len(srv.blob) - 1
                body = srv.blob[start:end + 1]
                self.send_response(206 if rng else 200)
                self.send_header("Content-Length", str(len(body)))
                if rng:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(srv.blob)}")
                self.end_headers()
                if len(body) > 1:
                    srv.bodies += 1
                    if srv.bodies <= srv.flaky:
                        self.wfile.write(body[:srv.cut])        # truncated: client sees IncompleteRead
                        self.wfile.flush()
                        self.connection.close()
                        return
                self.wfile.write(body)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/f"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)      # build_mix imports remote_zip as a top-level module too


def test_httpfile_read_retries_a_truncated_response():
    blob = bytes(range(256)) * 40
    s = _Server(blob)
    try:
        assert rz.HttpFile(s.url).read(100, 5000) == blob[100:5100]
    finally:
        s.close()


def test_concat_stream_resumes_a_part_cut_mid_body(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        data = b"x" * 3000
        ti = tarfile.TarInfo("gta_data/image/a.png")
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))
    blob = buf.getvalue()
    s = _Server(blob, flaky=1, cut=700)                       # cut inside the member's body
    try:
        got = bm.extract_tar_stream(io.BufferedReader(bm.ConcatStream([s.url]), 1 << 16), "gta_data/image",
                                    {"a.png"}, str(tmp_path))
    finally:
        s.close()
    assert got == {"a.png"} and (tmp_path / "a.png").read_bytes() == b"x" * 3000


def test_zero_byte_file_counts_as_not_fetched(tmp_path):
    """0-byte images left by an interrupted write must be fetched again, not skipped as present."""
    assert not rz.present(str(tmp_path / "gone.png"))
    (tmp_path / "empty.png").write_bytes(b"")
    (tmp_path / "ok.png").write_bytes(b"x")
    assert not rz.present(str(tmp_path / "empty.png")) and rz.present(str(tmp_path / "ok.png"))


def test_groundcua_download_retries(tmp_path):
    blob = b"\x89PNG" + b"p" * 4000
    s = _Server(blob, flaky=2, cut=50)
    try:
        bg._download(s.url, str(tmp_path / "x.png"))
    finally:
        s.close()
    assert (tmp_path / "x.png").read_bytes() == blob and not (tmp_path / "x.png.part").exists()
