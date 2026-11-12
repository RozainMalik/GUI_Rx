import pytest

from oai import pricing as p


def test_table_is_dated_and_luna_costs_what_the_spec_says():
    assert p.PRICES_AS_OF == "2026-09-16"
    assert p.PRICES["gpt-5.6-luna"] == (0.20, 0.02, 1.20) and p.PRICES["gpt-5.6-terra"] == (2.00, 0.20, 12.00)
    usage = {"input": 50_000, "cached": 30_000, "output": 4_000, "reasoning": 500}
    assert p.cost_usd(usage, "gpt-5.6-luna") == pytest.approx((20_000 * 0.20 + 30_000 * 0.02 + 4_000 * 1.20) / 1e6)
    assert p.cost_usd(usage, "gpt-5.6-terra") == pytest.approx(0.094)


def test_cached_tokens_never_exceed_input_and_missing_keys_are_zero():
    assert p.cost_usd({"input": 100, "cached": 500}, "gpt-5.6-luna") == pytest.approx(100 * 0.02 / 1e6)
    assert p.cost_usd({}, "gpt-5.6-luna") == 0.0


def test_dated_snapshots_resolve_by_prefix_and_unknown_models_raise():
    assert p.price_of("gpt-5.6-luna-2026-08-01") == p.PRICES["gpt-5.6-luna"]
    with pytest.raises(p.UnknownModel):
        p.price_of("gpt-9")
    with pytest.raises(p.UnknownModel):
        p.cost_usd({"input": 1}, "gpt-9")


def test_estimate_per_frame():
    assert p.estimate_usd_per_frame("gpt-5.6-luna") == 0.016
    assert p.estimate_usd_per_frame("gpt-5.6-terra-2026-08-01") == 0.16
    assert p.estimate_usd_per_frame("gpt-5.6-sol") == pytest.approx(0.016 * 20)


def test_batch_transport_halves_input_and_output_and_ignores_the_cache_discount():
    usage = {"input": 50_000, "cached": 30_000, "output": 4_000}
    assert p.cost_usd(usage, "gpt-5.6-luna", "batch") == pytest.approx((50_000 * 0.10 + 4_000 * 0.60) / 1e6)
    assert p.cost_usd(usage, "gpt-5.6-luna", "sync") == p.cost_usd(usage, "gpt-5.6-luna")
    with pytest.raises(ValueError, match="carrier-pigeon"):
        p.cost_usd(usage, "gpt-5.6-luna", "carrier-pigeon")
    assert p.TRANSPORTS == ("sync", "batch") and p.BATCH_DISCOUNT == 0.5 and p.CALLS_PER_FRAME == 19
    assert p.estimate_usd_per_call("gpt-5.6-luna") == pytest.approx(0.016 / 19)
    assert p.estimate_usd_per_call("gpt-5.6-luna", "batch") == pytest.approx(0.016 / 19 * 0.5)
