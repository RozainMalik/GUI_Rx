"""OpenAIGenerator: one Call -> one POST /v1/responses over httpx. Images first at
detail high, then one input_text holding the system and user prompts, so the images form an
identical - and therefore cached - prefix across every call on the same frame. Strict schema and
reasoning effort come from Call.kind; 429 backs off, 5xx retries, any other 4xx fails at once
with the body; usage lands in Reply.usage for the meter. No SDK: tests script an httpx transport."""
from __future__ import annotations

import asyncio

import httpx

from annotator.calling import Call, Reply, TransportError, image_to_data_url

from .schemas import text_format

BASE_URL = "https://api.openai.com/v1"
RATE_LIMIT_ATTEMPTS = 5
SERVER_ERROR_ATTEMPTS = 3
OUTPUT_TOKENS_NONE = 512          # effort none: the reply is the JSON and nothing else (as the Qwen path)
OUTPUT_TOKENS_REASONING = 4096    # any other effort: reasoning tokens share this budget
DEFAULT_EFFORT = "none"
EMPTY_ASSISTANT_TURN = "[no reply]"   # an empty assistant message is rejected by the API; a refusal repair needs a stand-in


def data_url_part(im) -> dict:
    return {"type": "input_image", "image_url": image_to_data_url(im), "detail": "high"}


def content_of(call: Call, image_part=None) -> list[dict]:
    """Images first (each through `image_part`, default an inline data URL), then one text part."""
    image_part = image_part or data_url_part
    parts = [image_part(im) for im in call.images]
    parts.append({"type": "input_text", "text": call.system + "\n\n" + call.user_text})
    return parts


def input_of(call: Call, image_part=None) -> list[dict]:
    messages = [{"role": "user", "content": content_of(call, image_part)}]
    for role, text in call.history:
        if role == "assistant":
            messages.append({"role": "assistant", "content": [{"type": "output_text", "text": text or EMPTY_ASSISTANT_TURN}]})
        else:
            messages.append({"role": "user", "content": [{"type": "input_text", "text": text}]})
    return messages


def request_body(call: Call, model: str, effort: str, image_part=None) -> dict:
    """One POST /v1/responses body - the generator's and the batch transport's."""
    body = {"model": model, "input": input_of(call, image_part), "reasoning": {"effort": effort},
            "max_output_tokens": OUTPUT_TOKENS_NONE if effort == "none" else OUTPUT_TOKENS_REASONING, "store": False}
    fmt = text_format(call.kind)
    if fmt is not None:
        body["text"] = {"format": fmt}
    return body


def parse_output(data) -> tuple[str, bool]:
    """(text, refused): every output_text part of every message item, joined. Refused when a
    refusal part was the only content - the caller treats it as an unreadable reply."""
    texts, saw_refusal = [], False
    for item in data.get("output") or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if part.get("type") == "output_text":
                texts.append(part.get("text") or "")
            elif part.get("type") == "refusal":
                saw_refusal = True
    text = "".join(texts)
    return text, saw_refusal and not text


def usage_of(data, model, effort, retries) -> dict:
    u = data.get("usage") or {}
    return {"input": int(u.get("input_tokens") or 0),
            "cached": int((u.get("input_tokens_details") or {}).get("cached_tokens") or 0),
            "output": int(u.get("output_tokens") or 0),
            "reasoning": int((u.get("output_tokens_details") or {}).get("reasoning_tokens") or 0),
            "retries": retries, "model": model, "effort": effort, "response_model": data.get("model") or model}


def retry_after(response, attempt) -> float:
    """The server's Retry-After when it sends one, else 1, 2, 4, 8 s by attempt."""
    try:
        return max(0.0, float(response.headers.get("Retry-After")))
    except (TypeError, ValueError):
        return float(2 ** (attempt - 1))


class OpenAIGenerator:
    def __init__(self, models, efforts, api_key, *, base_url=BASE_URL, concurrency=4, timeout=120.0, transport=None, sleep=asyncio.sleep):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is empty")
        self.models, self.efforts = dict(models), dict(efforts)
        self.sem = asyncio.Semaphore(concurrency)
        self.sleep = sleep
        self.client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=httpx.Timeout(timeout, connect=30.0),
                                        headers={"Authorization": f"Bearer {api_key}"}, transport=transport)
        self.calls = self.rate_limited = self.refusals = 0

    def model_for(self, kind) -> str:
        return self.models.get(kind) or self.models["ground"]

    def effort_for(self, kind) -> str:
        return self.efforts.get(kind, DEFAULT_EFFORT)

    def build_request(self, call: Call) -> dict:
        return request_body(call, self.model_for(call.kind), self.effort_for(call.kind))

    async def generate(self, call: Call) -> Reply:
        body = self.build_request(call)
        async with self.sem:
            self.calls += 1
            data, retries = await self._post(body)
        text, refused = parse_output(data)
        if refused:
            self.refusals += 1
        usage = usage_of(data, body["model"], body["reasoning"]["effort"], retries)
        return Reply(content=text, reasoning="", prompt_tokens=usage["input"], completion_tokens=usage["output"], usage=usage)

    async def _post(self, body) -> tuple[dict, int]:
        retries = rate_limits = server_errors = 0
        while True:
            try:
                r = await self.client.post("/responses", json=body)
            except httpx.HTTPError as e:                          # timeout, connection refused, ...
                server_errors += 1
                if server_errors >= SERVER_ERROR_ATTEMPTS:
                    raise TransportError(f"OpenAI unreachable after {server_errors} attempts: {e!r}") from e
                retries += 1
                await self.sleep(float(2 ** (server_errors - 1)))
                continue
            if r.status_code == 429:
                rate_limits += 1
                self.rate_limited += 1
                if rate_limits >= RATE_LIMIT_ATTEMPTS:
                    raise TransportError(f"OpenAI 429 after {rate_limits} attempts: {r.text[:500]}")
                retries += 1
                await self.sleep(retry_after(r, rate_limits))
                continue
            if r.status_code >= 500:
                server_errors += 1
                if server_errors >= SERVER_ERROR_ATTEMPTS:
                    raise TransportError(f"OpenAI {r.status_code} after {server_errors} attempts: {r.text[:500]}")
                retries += 1
                await self.sleep(float(2 ** (server_errors - 1)))
                continue
            if r.status_code >= 400:
                # a bad schema or a rejected image recurs on every frame: stop the batch at once, body attached
                raise TransportError(f"OpenAI {r.status_code}: {r.text[:500]}")
            try:
                return r.json(), retries
            except ValueError as e:
                raise TransportError(f"OpenAI reply is not JSON: {r.text[:200]}") from e

    async def aclose(self) -> None:
        await self.client.aclose()
