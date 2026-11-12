"""Pull exactly one JSON value out of a model reply, tolerating prose, code fences and a
leftover <think> block; and the message that asks the model to repair a bad reply."""
from __future__ import annotations

import json
import re

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_DECODER = json.JSONDecoder()

REPAIR_MESSAGE = (
    "Your previous reply could not be used: {error}\n"
    "Reply again with exactly one JSON value in the same schema that was requested, "
    "correcting that problem, and nothing else - no prose, no code fence.")


class ContractError(ValueError):
    """The reply holds no usable JSON value of the expected kind."""


def _bodies(text):
    text = _THINK.sub("", text or "")
    m = _FENCE.search(text)
    if m:
        yield m.group(1)
    yield text


def extract_json(text, kinds=("{", "[")):
    """The first balanced JSON value in `text` whose opening bracket is in `kinds`.
    Scans every opening bracket and tries a raw decode there, so prose before, after,
    or containing stray braces is tolerated."""
    for body in _bodies(text):
        for i, ch in enumerate(body):
            if ch not in kinds:
                continue
            try:
                value, _ = _DECODER.raw_decode(body, i)
            except json.JSONDecodeError:
                continue
            return value
    raise ContractError("no JSON value found in the reply")


def has_json(text) -> bool:
    try:
        extract_json(text)
        return True
    except ContractError:
        return False


def parse_object(text) -> dict:
    value = extract_json(text, kinds=("{",))
    if not isinstance(value, dict):
        raise ContractError("expected a JSON object")
    return value


def parse_array(text) -> list:
    """A bare JSON array - or, because strict output schemas need an object root, an object
    whose `boxes` key holds the array. The bare array is tried first so Qwen replies parse
    exactly as before."""
    try:
        value = extract_json(text, kinds=("[",))
    except ContractError:
        value = None
    if isinstance(value, list):
        return value
    try:
        obj = extract_json(text, kinds=("{",))
    except ContractError:
        raise ContractError("expected a JSON array") from None
    if isinstance(obj, dict) and isinstance(obj.get("boxes"), list):
        return obj["boxes"]
    raise ContractError("expected a JSON array")


def repair_message(error) -> str:
    return REPAIR_MESSAGE.format(error=str(error))
