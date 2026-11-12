"""The vLLM transport - the only module without tests, pure I/O. Posts the OpenAI
chat shape to /v1/chat/completions with chat_template_kwargs carrying the thinking switch,
retries transient failures with backoff, and raises TransportError when the server does not
answer - which leaves the stage undone so a rerun retries it."""
from __future__ import annotations

import asyncio

import httpx

from .calling import Call, Reply, TransportError, to_openai_messages


class VLLMGenerator:
    def __init__(self, base_url="http://localhost:8600/v1", model="qwen35_9b", timeout=900.0, retries=3):
        self.model = model
        self.retries = retries
        self.client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=httpx.Timeout(timeout, connect=30.0))

    async def generate(self, call: Call) -> Reply:
        body = {"model": self.model, "messages": to_openai_messages(call), "temperature": 0,
                "max_tokens": call.max_tokens, "chat_template_kwargs": {"enable_thinking": call.thinking}}
        last = None
        for attempt in range(self.retries):
            try:
                r = await self.client.post("/chat/completions", json=body)
                r.raise_for_status()
                data = r.json()
                msg = data["choices"][0]["message"]
                usage = data.get("usage") or {}
                # vLLM's reasoning parser names the field reasoning_content (older) or reasoning (newer).
                return Reply(content=msg.get("content") or "",
                             reasoning=msg.get("reasoning_content") or msg.get("reasoning") or "",
                             prompt_tokens=int(usage.get("prompt_tokens") or 0),
                             completion_tokens=int(usage.get("completion_tokens") or 0))
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if 400 <= status < 500:
                    # A bad chat_template_kwargs or an undecodable image recurs identically on
                    # every retry and every rerun; the body is the only diagnostic - fail fast.
                    raise TransportError(f"vLLM {status}: {e.response.text[:500]}") from e
                last = e                                # 5xx: transient, keep retrying
            except (httpx.HTTPError, KeyError, ValueError, IndexError) as e:
                last = e
            if attempt < self.retries - 1:              # no sleep after the final attempt
                await asyncio.sleep(2 ** attempt)
        detail = ""
        if isinstance(last, httpx.HTTPStatusError):
            detail = f" (vLLM {last.response.status_code}: {last.response.text[:500]})"
        raise TransportError(f"vLLM call failed after {self.retries} attempts: {last!r}{detail}")

    async def models(self) -> list[str]:
        r = await self.client.get("/models")
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]

    async def aclose(self) -> None:
        await self.client.aclose()
