"""Shared test doubles. No network, no key: the model is FakeOpenAI behind an httpx MockTransport."""
import asyncio
import json
import re
import textwrap
from collections import Counter
from pathlib import Path

import httpx
import pytest
from PIL import Image, ImageDraw

FRAME_W, FRAME_H = 1920, 1080


def make_frame(width=FRAME_W, height=FRAME_H):
    """A synthetic screenshot: grey chrome, a white 'menu' panel at (200,100)-(900,500) with dark rows."""
    img = Image.new("RGB", (width, height), (60, 60, 64))
    d = ImageDraw.Draw(img)
    d.rectangle((200, 100, 900, 500), fill=(245, 245, 245))
    for y in range(120, 500, 40):
        d.rectangle((210, y, 890, y + 24), fill=(200, 200, 210))
    return img


@pytest.fixture
def frame():
    return make_frame()


def run(coro):
    """Synchronous entry for async code under test (no pytest-asyncio)."""
    return asyncio.run(coro)


def make_data(root, apps=("Alpha", "Beta"), n=2) -> Path:
    """<root>/<App>/<App>_NNN.png for each app. Every frame differs in a corner mark, as real
    screenshots do - the batch transport keys requests by image content."""
    root = Path(root)
    for app in apps:
        folder = root / app
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(1, n + 1):
            img = make_frame()
            shade = (sum(map(ord, app)) * 31 + i * 7) % 200
            ImageDraw.Draw(img).rectangle((0, 0, 6, 6), fill=(shade, shade, 255 - shade))
            img.save(folder / f"{app}_{i:03d}.png")
    return root


def write_config(tmp_path, **overrides) -> Path:
    """A config.toml under tmp_path whose paths all live under tmp_path. Overrides are TOML
    fragments keyed by section, e.g. write_config(tmp, batches='sizes = [2, 2]\\ngate = 0.5')."""
    tmp_path = Path(tmp_path)
    data = str(tmp_path / "data").replace("\\", "/")
    sections = {
        "models": "\n".join(f'{k} = "gpt-5.6-luna"' for k in ("propose", "describe", "ground", "verify", "refine_ground")),
        "effort": 'propose = "low"\ndescribe = "low"\nground = "none"\nverify = "none"\nrefine_ground = "none"',
        "run": 'coord_mode = "pixels"\nconcurrency = 2\nframe_concurrency = 2\nseed = 7\nearly_stop = false',
        "batches": "sizes = [2, 2]\ngate = 0.75\nexpected_apps = 2",
        "caps": "batch_usd = 0\nrun_usd = 50",
        "paths": f'data = "{data}"\nbenchmark = "benchmark"\nwork = "work"\nexport = "export"\nreports = "reports"\nimage_root = "{data}"',
        "remote": 'login = "user@host"\ndata = "remote/data"\nbenchmark = "MedicalDataset"',
    }
    sections.update(overrides)
    text = "\n\n".join(f"[{name}]\n{body}" for name, body in sections.items())
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.toml"
    path.write_text(textwrap.dedent(text) + "\n", encoding="utf-8")
    return path


INSTR = {"functional": "Start a slideshow of the loaded images.",
         "visual": "Click the menu item labelled 'Slideshow...'.",
         "spatial": "Click the eighth row of the open File menu, just below 'Thumbnails'."}
REGION_PX, TARGET_PX = (192, 108, 960, 432), (288, 162, 576, 216)

GOOD_REPLIES = {
    "propose": {"intent": "open the File menu", "focus_region": {"bbox": list(REGION_PX), "kind": "menu", "why": "w"},
                "candidates": [{"rank": 1, "element_name": "Slideshow...", "element_kind": "menu item", "text_on_screen": "Slideshow...",
                                "icon": "", "state": "highlighted", "bbox": list(TARGET_PX)}]},
    "describe": {"element_present": True, "bbox": [96, 54, 384, 108], "text_on_screen": "Slideshow...", "icon": "",
                 "neighbours": {"above": "Thumbnails", "below": "Batch", "left": "", "right": ""},
                 "position": "8th of 24 rows", "instructions": dict(INSTR)},
    "ground": {"boxes": [{"bbox_2d": list(TARGET_PX), "label": "Slideshow..."}]},
    "refine_ground": {"boxes": []},
    "verify": {"text_in_box": "Slideshow...", "element_in_box": "row", "verdict": "correct", "reason": "r", "corrected_bbox": None},
}
BAD_REPLIES = {
    "propose": [{"intent": "i", "candidates": []},                                              # focus_region missing
                {**GOOD_REPLIES["propose"], "focus_region": {"bbox": [0, 0, 1, 1], "kind": "window", "why": ""}}],   # bad enum
    "describe": [{**GOOD_REPLIES["describe"], "extra": 1},                                     # additional property
                 {**GOOD_REPLIES["describe"], "instructions": {"functional": "f", "visual": "v"}}],   # spatial missing
    "ground": [[{"bbox_2d": [1, 2, 3, 4], "label": "x"}],                                      # bare array, not an object
               {"boxes": [{"bbox_2d": "1,2,3,4", "label": "x"}]}],                             # bbox is a string
    "verify": [{**GOOD_REPLIES["verify"], "verdict": "maybe"},
               {"text_in_box": "", "element_in_box": "", "verdict": "correct", "reason": ""}],   # corrected_bbox missing
}


def response_json(text, *, input_tokens=2448, cached=0, output_tokens=40, reasoning=0, model="gpt-5.6-luna", refusal=None) -> dict:
    """A Responses-API body: a reasoning item, then a message with output_text (or a refusal)."""
    content = [{"type": "refusal", "refusal": refusal}] if refusal else [{"type": "output_text", "text": text, "annotations": []}]
    return {"id": "resp_1", "object": "response", "model": model,
            "output": [{"type": "reasoning", "id": "rs_1", "summary": []},
                       {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": content}],
            "usage": {"input_tokens": input_tokens, "input_tokens_details": {"cached_tokens": cached},
                      "output_tokens": output_tokens, "output_tokens_details": {"reasoning_tokens": reasoning},
                      "total_tokens": input_tokens + output_tokens}}


def user_text_of(body) -> str:
    return body["input"][0]["content"][-1]["text"]


class FakeOpenAI:
    """A scripted Responses API behind httpx.MockTransport. Either `replies` consumed in order
    or a `router(body) -> reply`. A reply is a str (200 with that text), a dict (the whole body),
    a (status, body, headers) tuple, or an httpx.HTTPError to raise. Every decoded request body
    is kept in `.requests`, every header set in `.headers`."""

    def __init__(self, replies=None, router=None, input_tokens=2448, cached=0, output_tokens=40, reasoning=0):
        self.router, self.replies = router, list(replies or [])
        self.usage = dict(input_tokens=input_tokens, cached=cached, output_tokens=output_tokens, reasoning=reasoning)
        self.requests, self.headers = [], []

    def transport(self):
        return httpx.MockTransport(self._handle)

    def _handle(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        self.headers.append(dict(request.headers))
        if self.router is not None:
            reply = self.router(body)
        else:
            if not self.replies:
                raise AssertionError(f"FakeOpenAI ran out of replies; last text:\n{user_text_of(body)[:200]}")
            reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, tuple):
            status, payload, headers = reply
            return httpx.Response(status, json=payload, headers=headers, request=request)
        if isinstance(reply, str):
            reply = response_json(reply, **self.usage)
        return httpx.Response(200, json=reply, request=request)


async def nosleep(seconds):
    nosleep.calls.append(seconds)


nosleep.calls = []


@pytest.fixture(autouse=True)
def _reset_nosleep():
    nosleep.calls.clear()


PROPOSE_MARK, DESCRIBE_MARK, GROUND_MARK, VERIFY_MARK, REFINE_MARK, CALIBRATE_MARK = (
    "Step 1 - Intent", "ZOOMED CROP", "Locate the element in this screenshot", "red rectangle", "magnified region",
    "Describe this screenshot")


def happy_router(mode="pixels", calibrate_box=(200, 110, 380, 160)):
    """Walks one 1920x1080 frame to tier A in the Responses body shape: one candidate, every
    grounding agreeing, the judge saying correct, no refine. Calibration steps ('Click step N')
    are boxed inside the benchmark ground truth unless the instruction says 'miss'."""
    from annotator.geometry import full_space
    from annotator.imaging import make_crop, region_window
    full = full_space(FRAME_W, FRAME_H, mode)
    crop_space = make_crop(make_frame(), region_window(REGION_PX, FRAME_W, FRAME_H, 0.05)).space(mode)

    def route(body):
        t = user_text_of(body)
        if PROPOSE_MARK in t:
            cand = {**GOOD_REPLIES["propose"]["candidates"][0], "bbox": full.from_px(TARGET_PX)}
            return json.dumps({**GOOD_REPLIES["propose"], "focus_region": {"bbox": full.from_px(REGION_PX), "kind": "menu", "why": "w"},
                               "candidates": [cand]})
        if DESCRIBE_MARK in t:
            return json.dumps({**GOOD_REPLIES["describe"], "bbox": crop_space.from_px(TARGET_PX)})
        if GROUND_MARK in t:
            if "Click step" in t:
                box = (600, 600, 700, 650) if "miss" in t else calibrate_box
                return json.dumps({"boxes": [{"bbox_2d": full.from_px(box), "label": "x"}]})
            return json.dumps({"boxes": [{"bbox_2d": full.from_px(TARGET_PX), "label": "x"}]})
        if VERIFY_MARK in t:
            return json.dumps(GOOD_REPLIES["verify"])
        if REFINE_MARK in t:
            return json.dumps({"boxes": []})
        if CALIBRATE_MARK in t:
            return "screenshot"
        raise AssertionError("unexpected call: " + t[:80])
    return route


def parse_multipart(content: bytes, content_type: str) -> dict[str, tuple[str | None, bytes]]:
    """{field: (filename or None, bytes)} of a multipart/form-data body - enough for httpx's own encoding."""
    boundary = content_type.split("boundary=", 1)[1].strip().encode()
    fields = {}
    for part in content.split(b"--" + boundary)[1:]:
        if part.strip() in (b"", b"--"):
            continue
        head, _, body = part.partition(b"\r\n\r\n")
        if body.endswith(b"\r\n"):
            body = body[:-2]
        name = re.search(rb'name="([^"]+)"', head).group(1).decode()
        fn = re.search(rb'filename="([^"]+)"', head)
        fields[name] = (fn.group(1).decode() if fn else None, body)
    return fields


class FakeBatchOpenAI:
    """The Files and Batches endpoints behind httpx.MockTransport. A job is answered
    at creation: every line's body goes through `router(body)` exactly as FakeOpenAI does (str ->
    200 text, dict -> that body, (status, body, headers) -> a non-200 line); `errors(body) -> str|None`
    puts the line in the error file instead. GET /batches/{id} answers `in_progress` until the
    `polls_before_done`-th poll, then `completed` - or `expired` with the last `expire_last` lines of
    the FIRST job unanswered, or `failed` with `fail` as the message. `create_ok_before_failing=n`
    makes the (n+1)-th POST /batches raise a connection error `create_failures` times. A POST
    /responses is an AssertionError: the batch transport never calls it."""

    def __init__(self, router, *, errors=None, polls_before_done=1, expire_last=0, fail=None,
                 create_ok_before_failing=None, create_failures=3, poll_failures=0, input_tokens=2448, output_tokens=40):
        self.router, self.errors, self.polls_before_done, self.expire_last, self.fail = router, errors, polls_before_done, expire_last, fail
        self.create_ok_before_failing, self.create_failures = create_ok_before_failing, create_failures
        self.poll_failures = poll_failures                      # the first N GET /batches/{id} raise a connection error
        self.usage = dict(input_tokens=input_tokens, output_tokens=output_tokens)
        self.files, self.batches, self.requests, self.custom_ids = {}, {}, [], []
        self.polls, self.cancelled, self.created = Counter(), [], 0

    def transport(self):
        return httpx.MockTransport(self._handle)

    def _handle(self, request):
        path, method = request.url.path.removeprefix("/v1"), request.method
        if path == "/responses":
            raise AssertionError("sync /responses call under the batch transport")
        if method == "POST" and path == "/files":
            return self._upload(request)
        if method == "POST" and path == "/batches":
            return self._create(request)
        if (m := re.fullmatch(r"/batches/([^/]+)", path)) and method == "GET":
            if self.poll_failures > 0:
                self.poll_failures -= 1
                raise httpx.ConnectError("[Errno 11001] getaddrinfo failed", request=request)
            return httpx.Response(200, json=self._poll(m.group(1)), request=request)
        if (m := re.fullmatch(r"/batches/([^/]+)/cancel", path)) and method == "POST":
            self.batches[m.group(1)]["status"] = "cancelled"
            self.cancelled.append(m.group(1))
            return httpx.Response(200, json=self._public(self.batches[m.group(1)]), request=request)
        if (m := re.fullmatch(r"/files/([^/]+)/content", path)) and method == "GET":
            return httpx.Response(200, content=self.files[m.group(1)]["content"], request=request)
        return httpx.Response(404, json={"error": {"message": f"no route {method} {path}"}}, request=request)

    def _upload(self, request):
        fields = parse_multipart(request.content, request.headers["content-type"])
        fid = f"file-{len(self.files) + 1}"
        name, content = fields["file"]
        form = {k: v[1].decode() for k, v in fields.items() if k != "file"}
        self.files[fid] = {"name": name, "purpose": form.get("purpose"), "content": content, "form": form,
                           "timeout": dict(request.extensions.get("timeout") or {})}
        expires = int(form["expires_after[seconds]"]) if "expires_after[seconds]" in form else None
        return httpx.Response(200, json={"id": fid, "object": "file", "purpose": form.get("purpose"), "bytes": len(content),
                                         "filename": name, "expires_at": 1_800_000_000 + expires if expires else None}, request=request)

    def _create(self, request):
        if self.create_ok_before_failing is not None and self.created >= self.create_ok_before_failing and self.create_failures > 0:
            self.create_failures -= 1
            raise httpx.ConnectError("boom", request=request)
        body = json.loads(request.content)
        lines = self.files[body["input_file_id"]]["content"].decode("utf-8").splitlines()
        self.created += 1
        jid = f"batch_{self.created}"
        expired = self.created == 1 and self.expire_last > 0
        out, err = [], []
        for i, line in enumerate(lines):
            req = json.loads(line)
            self.requests.append(req["body"])
            self.custom_ids.append(req["custom_id"])
            if expired and i >= len(lines) - self.expire_last:
                continue
            msg = self.errors(req["body"]) if self.errors else None
            if msg:
                err.append({"id": f"batch_req_{i}", "custom_id": req["custom_id"], "response": None,
                            "error": {"code": "invalid_request", "message": msg}})
                continue
            reply = self.router(req["body"])
            if isinstance(reply, tuple):
                status, payload, _ = reply
                out.append({"id": f"batch_req_{i}", "custom_id": req["custom_id"], "error": None,
                            "response": {"status_code": status, "request_id": f"req_{i}", "body": payload}})
                continue
            if isinstance(reply, str):
                reply = response_json(reply, **self.usage)
            out.append({"id": f"batch_req_{i}", "custom_id": req["custom_id"], "error": None,
                        "response": {"status_code": 200, "request_id": f"req_{i}", "body": reply}})
        out_id = err_id = None
        if out:
            out_id = f"file-out-{jid}"
            self.files[out_id] = {"name": "out.jsonl", "purpose": "batch_output", "content": "".join(json.dumps(l) + "\n" for l in out).encode(), "form": {}}
        if err:
            err_id = f"file-err-{jid}"
            self.files[err_id] = {"name": "err.jsonl", "purpose": "batch_output", "content": "".join(json.dumps(l) + "\n" for l in err).encode(), "form": {}}
        self.batches[jid] = {"id": jid, "object": "batch", "endpoint": body["endpoint"], "input_file_id": body["input_file_id"],
                             "completion_window": body["completion_window"], "metadata": body.get("metadata"), "status": "validating",
                             "output_file_id": None, "error_file_id": None, "errors": None,
                             "request_counts": {"total": len(lines), "completed": 0, "failed": 0},
                             "_out": out_id, "_err": err_id, "_expired": expired, "_done": len(out), "_failed": len(err)}
        return httpx.Response(200, json=self._public(self.batches[jid]), request=request)

    def _poll(self, jid):
        b = self.batches[jid]
        self.polls[jid] += 1
        if b["status"] in ("validating", "in_progress"):
            b["status"] = "in_progress"
            if self.polls[jid] >= self.polls_before_done:
                if self.fail:
                    b["status"], b["errors"] = "failed", {"object": "list", "data": [{"code": "invalid_json_line", "line": 1, "message": self.fail}]}
                else:
                    b["status"] = "expired" if b["_expired"] else "completed"
                    b["output_file_id"], b["error_file_id"] = b["_out"], b["_err"]
                    b["request_counts"] = {"total": b["request_counts"]["total"], "completed": b["_done"], "failed": b["_failed"]}
        return self._public(b)

    @staticmethod
    def _public(b):
        return {k: v for k, v in b.items() if not k.startswith("_")}
