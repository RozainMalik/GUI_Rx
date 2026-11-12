import json

import httpx
import pytest

from annotator.calling import TransportError
from oai.batchclient import BatchClient
from tests.conftest import FakeBatchOpenAI, happy_router, parse_multipart


def client_for(fake, **kw):
    return BatchClient("sk-test-1234", transport=fake.transport(), sleep=lambda s: None, **kw)


def test_upload_sends_purpose_expiry_and_the_bytes_as_multipart():
    fake = FakeBatchOpenAI(router=happy_router())
    obj = client_for(fake).upload("Alpha_001.png", b"\x89PNG\r\n\x1a\nstuff\r\n", "vision", expires_seconds=2_592_000)
    assert obj["id"] == "file-1" and obj["purpose"] == "vision" and obj["expires_at"]
    f = fake.files["file-1"]
    assert f["name"] == "Alpha_001.png" and f["content"] == b"\x89PNG\r\n\x1a\nstuff\r\n"          # trailing CRLF kept
    assert f["form"] == {"purpose": "vision", "expires_after[anchor]": "created_at", "expires_after[seconds]": "2592000"}
    obj = client_for(fake).upload("round1.part1.jsonl", b'{"custom_id": "x"}\n', "batch")
    assert fake.files[obj["id"]]["form"] == {"purpose": "batch"}


def test_create_poll_download_and_cancel_round_trip():
    fake = FakeBatchOpenAI(router=happy_router(), polls_before_done=2)
    c = client_for(fake)
    line = {"custom_id": "abc", "method": "POST", "url": "/v1/responses",
            "body": {"model": "gpt-5.6-luna", "input": [{"role": "user", "content": [{"type": "input_text", "text": "Step 1 - Intent"}]}]}}
    fid = c.upload("p.jsonl", (json.dumps(line) + "\n").encode(), "batch")["id"]
    job = c.create_batch(fid, {"batch": "9", "round": "1", "part": "p.jsonl"})
    assert job["status"] == "validating" and job["endpoint"] == "/v1/responses" and job["completion_window"] == "24h"
    assert job["metadata"] == {"batch": "9", "round": "1", "part": "p.jsonl"} and fake.requests[0] == line["body"]
    assert c.get_batch(job["id"])["status"] == "in_progress"
    done = c.get_batch(job["id"])
    assert done["status"] == "completed" and done["request_counts"] == {"total": 1, "completed": 1, "failed": 0}
    out = c.download(done["output_file_id"]).decode().splitlines()
    assert len(out) == 1 and json.loads(out[0])["custom_id"] == "abc" and json.loads(out[0])["response"]["status_code"] == 200
    assert c.cancel_batch(job["id"])["status"] == "cancelled" and fake.cancelled == [job["id"]]


def test_5xx_retries_three_times_then_fails_and_4xx_fails_fast():
    calls = []

    def flaky(request):
        calls.append(request.url.path)
        return httpx.Response(503, json={"error": {"message": "down"}}, request=request)
    c = BatchClient("sk-test", transport=httpx.MockTransport(flaky), sleep=lambda s: None)
    with pytest.raises(TransportError, match="503 after 3 attempts"):
        c.get_batch("batch_1")
    assert len(calls) == 3
    c = BatchClient("sk-test", transport=httpx.MockTransport(lambda r: httpx.Response(400, json={"error": {"message": "bad file"}}, request=r)), sleep=lambda s: None)
    with pytest.raises(TransportError, match="400 .*bad file"):
        c.create_batch("file-1", {})


def test_parse_multipart_keeps_binary_bodies_intact():
    body = (b"--b\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nvision\r\n"
            b"--b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.png\"\r\nContent-Type: image/png\r\n\r\n\x00\r\n\x01\r\n--b--\r\n")
    fields = parse_multipart(body, "multipart/form-data; boundary=b")
    assert fields == {"purpose": (None, b"vision"), "file": ("a.png", b"\x00\r\n\x01")}


def test_upload_write_timeout_scales_with_the_payload():
    from oai.batchclient import MIN_UPLOAD_BYTES_PER_S
    fake = FakeBatchOpenAI(router=happy_router())
    c = client_for(fake)
    c.upload("small.jsonl", b"x" * 1000, "batch")
    c.upload("big.jsonl", b"x" * 50_000_000, "batch")
    small, big = fake.files["file-1"]["timeout"], fake.files["file-2"]["timeout"]
    assert small["write"] == 120.0 and small["read"] == 120.0 and small["connect"] == 30.0
    assert big["write"] >= 60 + 50_000_000 / MIN_UPLOAD_BYTES_PER_S and big["read"] == 120.0
