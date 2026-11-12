"""The price table, dated, and the one cost formula.
Changing a price is one line here; `status` re-prices the whole ledger from it. Reasoning tokens
bill as output and are already counted inside `output`, so they never enter the formula twice.
The Batch API bills half of everything; whether it honours the cached-input discount is
undocumented, so batch rows price cached tokens as plain input."""
from __future__ import annotations

PRICES_AS_OF = "2026-09-16"
# USD per 1M tokens: (input, cached input, output)
PRICES = {
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gpt-5.6-terra": (2.00, 0.20, 12.00),
    "gpt-5.6-sol": (4.00, 0.40, 20.00),
}
# USD per frame before the meter has measured anything.
ESTIMATES = {"gpt-5.6-luna": 0.016, "gpt-5.6-terra": 0.16}
TRANSPORTS = ("sync", "batch")
BATCH_DISCOUNT = 0.5
CALLS_PER_FRAME = 19          # measured while building the dataset: 27,676 calls / ~1,470 frames


class UnknownModel(KeyError):
    """No price for this model id - add it to PRICES before spending on it."""


def _lookup(table, model):
    if model in table:
        return table[model]
    for key in sorted(table, key=len, reverse=True):       # dated snapshot: gpt-5.6-luna-2026-08-01
        if str(model).startswith(key):
            return table[key]
    raise UnknownModel(model)


def price_of(model) -> tuple[float, float, float]:
    return _lookup(PRICES, model)


def cost_usd(usage: dict, model: str, transport: str = "sync") -> float:
    p_in, p_cached, p_out = price_of(model)
    if transport == "batch":
        p_in, p_cached, p_out = p_in * BATCH_DISCOUNT, p_in * BATCH_DISCOUNT, p_out * BATCH_DISCOUNT
    elif transport != "sync":
        raise ValueError(f"unknown transport {transport!r}; one of {', '.join(TRANSPORTS)}")
    inp = int(usage.get("input") or 0)
    cached = min(int(usage.get("cached") or 0), inp)
    out = int(usage.get("output") or 0)
    return round(((inp - cached) * p_in + cached * p_cached + out * p_out) / 1_000_000, 8)


def estimate_usd_per_frame(model) -> float:
    try:
        return _lookup(ESTIMATES, model)
    except UnknownModel:
        # scale luna's estimate by the input price ratio - input tokens are ~2/3 of a frame's cost
        return round(ESTIMATES["gpt-5.6-luna"] * price_of(model)[0] / PRICES["gpt-5.6-luna"][0], 6)


def estimate_usd_per_call(model, transport: str = "sync") -> float:
    """The no-ledger fallback for a round's projection."""
    per_call = estimate_usd_per_frame(model) / CALLS_PER_FRAME
    return per_call * BATCH_DISCOUNT if transport == "batch" else per_call
