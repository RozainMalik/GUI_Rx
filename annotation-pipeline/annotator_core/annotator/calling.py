"""The call discipline every stage shares: a Call, a Reply, the Generator
protocol, the OpenAI message shape, the thinking-exhausted fallback and the single repair
replay. Stages never talk to the transport directly; they go through call_validated()."""
from __future__ import annotations

import base64
import dataclasses
import io
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .contract import ContractError, has_json, repair_message
from .validators import SoftValidationError, ValidationError

THINKING_MAX_TOKENS = 4096
PLAIN_MAX_TOKENS = 512


@dataclass(frozen=True)
class Call:
    system: str
    user_text: str
    images: tuple = ()          # PIL images in the order the prompt names them (Image 1, Image 2)
    thinking: bool = False
    history: tuple = ()         # ((role, text), ...) appended after the user turn - the repair replay
    kind: str = ""              # propose | describe | ground | verify | refine_ground | probe - set by the stage;
                                 # a provider client picks its output schema and effort by it, a ledger attributes cost

    @property
    def max_tokens(self) -> int:
        return THINKING_MAX_TOKENS if self.thinking else PLAIN_MAX_TOKENS

    def with_thinking(self, on: bool) -> "Call":
        return dataclasses.replace(self, thinking=on)

    def with_repair(self, raw: str, error) -> "Call":
        return dataclasses.replace(self, history=self.history + (("assistant", raw), ("user", repair_message(error))))


@dataclass
class Reply:
    content: str
    reasoning: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usage: dict | None = None   # a provider's token accounting: input, cached, output, reasoning (+ whatever it adds)


class Generator(Protocol):
    async def generate(self, call: Call) -> Reply: ...


class TransportError(RuntimeError):
    """The server did not answer. The stage stays undone so a rerun retries exactly it."""


def image_to_data_url(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def to_openai_messages(call: Call) -> list[dict]:
    content = [{"type": "image_url", "image_url": {"url": image_to_data_url(im)}} for im in call.images]
    content.append({"type": "text", "text": call.user_text})
    messages = [{"role": "system", "content": call.system}, {"role": "user", "content": content}]
    for role, text in call.history:
        messages.append({"role": role, "content": text})
    return messages


@dataclass
class Exchange:
    value: Any                  # what interpret() returned, or None when unreadable
    raw: str                    # the last raw content, kept verbatim in the record
    calls: int
    thinking_chars: int
    thinking_fallback: bool
    repaired: bool
    error: str | None
    soft_errors: list = field(default_factory=list)


async def call_validated(gen: Generator, call: Call, interpret: Callable[[str], Any]) -> Exchange:
    """generate -> interpret(raw) (parse + validate; raises ContractError / ValidationError) ->
    on failure one repair replay -> on a second failure value=None with the error.
    A thinking-on reply whose content holds no JSON (the budget went to reasoning) is retried
    once with thinking off, counted as a fallback - both on the first reply and, if it recurs,
    on the repair reply too (the repair call otherwise inherits call.thinking)."""
    calls, thinking_chars, fallback = 0, 0, False
    reply = await gen.generate(call)
    calls += 1
    thinking_chars += len(reply.reasoning or "")
    if call.thinking and not has_json(reply.content):
        call = call.with_thinking(False)
        reply = await gen.generate(call)
        calls += 1
        fallback = True
    raw = reply.content or ""
    try:
        return Exchange(interpret(raw), raw, calls, thinking_chars, fallback, False, None)
    except (ContractError, ValidationError) as first:
        repair_call = call.with_repair(raw, first)
        reply = await gen.generate(repair_call)
        calls += 1
        thinking_chars += len(reply.reasoning or "")
        if call.thinking and not has_json(reply.content):
            repair_call = repair_call.with_thinking(False)
            reply = await gen.generate(repair_call)
            calls += 1
            fallback = True
        raw = reply.content or ""
        try:
            return Exchange(interpret(raw), raw, calls, thinking_chars, fallback, True, None)
        except SoftValidationError as soft:
            return Exchange(soft.value, raw, calls, thinking_chars, fallback, True, None, list(soft.errors))
        except (ContractError, ValidationError) as second:
            return Exchange(None, raw, calls, thinking_chars, fallback, True, str(second))
