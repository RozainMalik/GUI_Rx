import asyncio
import json

import httpx
import pytest

from annotator.calling import Call, TransportError
from oai import client as oc
from tests.conftest import FakeOpenAI, make_frame, nosleep, response_json, run, user_text_of

MODELS = {k: "gpt-5.6-luna" for k in ("propose", "describe", "ground", "verify", "refine_ground")}
MODELS["describe"] = "gpt-5.6-terra"
EFFORTS = {"propose": "low", "describe": "low", "ground": "none", "verify": "none", "refine_ground": "none"}


def gen_for(fake, **kw):
    return oc.OpenAIGenerator(MODELS, EFFORTS, "sk-test-1234", transport=fake.transport(), sleep=nosleep, **kw)


def test_request_shape_images_first_high_detail_schema_and_effort_by_kind(frame):
    fake = FakeOpenAI(['{"boxes": []}'])
    call = Call(system="SYS", user_text="USER", images=(frame, make_frame(100, 50)), thinking=False, kind="refine_ground")
    reply = run(gen_for(fake).generate(call))
    body = fake.requests[0]
    content = body["input"][0]["content"]
    assert body["input"][0]["role"] == "user" and len(content) == 3
    assert content[0]["type"] == "input_image" and content[0]["detail"] == "high" and content[0]["image_url"].startswith("data:image/png;base64,")
    assert content[1]["type"] == "input_image" and content[2] == {"type": "input_text", "text": "SYS\n\nUSER"}
    assert body["model"] == "gpt-5.6-luna" and body["reasoning"] == {"effort": "none"} and body["max_output_tokens"] == 512
    assert body["text"]["format"]["name"] == "annotator_refine_ground" and body["text"]["format"]["strict"] is True
    assert body["store"] is False and fake.headers[0]["authorization"] == "Bearer sk-test-1234"
    assert reply.content == '{"boxes": []}'


def test_describe_uses_its_own_model_and_reasoning_budget(frame):
    fake = FakeOpenAI(["{}"])
    run(gen_for(fake).generate(Call(system="s", user_text="u", images=(frame,), thinking=True, kind="describe")))
    body = fake.requests[0]
    assert body["model"] == "gpt-5.6-terra" and body["reasoning"] == {"effort": "low"} and body["max_output_tokens"] == 4096


def test_probe_kind_has_no_schema_and_falls_back_to_the_ground_model(frame):
    fake = FakeOpenAI(["screenshot"])
    reply = run(gen_for(fake).generate(Call(system="s", user_text="Describe this screenshot in one word.", images=(frame,), kind="probe")))
    assert "text" not in fake.requests[0] and fake.requests[0]["model"] == "gpt-5.6-luna" and fake.requests[0]["reasoning"] == {"effort": "none"}
    assert reply.prompt_tokens == 2448


def test_repair_history_maps_to_assistant_and_user_turns():
    fake = FakeOpenAI(["{}"])
    call = Call(system="s", user_text="u", kind="verify").with_repair("bad raw", "the error")
    run(gen_for(fake).generate(call))
    msgs = fake.requests[0]["input"]
    assert msgs[1] == {"role": "assistant", "content": [{"type": "output_text", "text": "bad raw"}]}
    assert msgs[2]["role"] == "user" and msgs[2]["content"][0]["type"] == "input_text" and "the error" in msgs[2]["content"][0]["text"]
    empty = Call(system="s", user_text="u", kind="verify").with_repair("", "refused")
    run(gen_for(FakeOpenAI(["{}"])).generate(empty))       # an empty assistant turn is not sent as such


def test_usage_lands_in_the_reply():
    fake = FakeOpenAI([response_json("{}", input_tokens=5000, cached=1024, output_tokens=700, reasoning=500, model="gpt-5.6-luna-2026-08-01")])
    reply = run(gen_for(fake).generate(Call(system="s", user_text="u", kind="propose")))
    assert reply.usage == {"input": 5000, "cached": 1024, "output": 700, "reasoning": 500, "retries": 0,
                           "model": "gpt-5.6-luna", "effort": "low", "response_model": "gpt-5.6-luna-2026-08-01"}
    assert reply.prompt_tokens == 5000 and reply.completion_tokens == 700 and reply.reasoning == ""


def test_429_honours_retry_after_then_backs_off_and_counts_retries():
    fake = FakeOpenAI([(429, {"error": "slow down"}, {"Retry-After": "3"}), (429, {}, {}), (429, {}, {}), "ok"])
    gen = gen_for(fake)
    reply = run(gen.generate(Call(system="s", user_text="u", kind="ground")))
    assert reply.content == "ok" and reply.usage["retries"] == 3 and gen.rate_limited == 3 and gen.calls == 1
    assert nosleep.calls == [3.0, 2.0, 4.0]


def test_429_five_times_is_a_transport_error():
    fake = FakeOpenAI([(429, {}, {})] * 5 + ["never"])
    with pytest.raises(TransportError, match="429"):
        run(gen_for(fake).generate(Call(system="s", user_text="u", kind="ground")))
    assert len(fake.requests) == 5 and nosleep.calls == [1.0, 2.0, 4.0, 8.0]


def test_5xx_and_connection_errors_retry_three_times_then_fail():
    fake = FakeOpenAI([(503, {"error": "busy"}, {}), httpx.ConnectError("down"), (500, {}, {})])
    with pytest.raises(TransportError, match="after 3 attempts"):
        run(gen_for(fake).generate(Call(system="s", user_text="u", kind="ground")))
    assert len(fake.requests) == 3 and nosleep.calls == [1.0, 2.0]
    fake = FakeOpenAI([(502, {}, {}), "ok"])
    assert run(gen_for(fake).generate(Call(system="s", user_text="u", kind="ground"))).usage["retries"] == 1


def test_4xx_fails_fast_with_the_body():
    fake = FakeOpenAI([(400, {"error": {"message": "invalid schema: boxes"}}, {}), "never"])
    with pytest.raises(TransportError, match="invalid schema: boxes"):
        run(gen_for(fake).generate(Call(system="s", user_text="u", kind="ground")))
    assert len(fake.requests) == 1 and nosleep.calls == []


def test_refusal_is_an_empty_reply_and_counted():
    fake = FakeOpenAI([response_json("", refusal="I can't help with that.")])
    gen = gen_for(fake)
    reply = run(gen.generate(Call(system="s", user_text="u", kind="describe")))
    assert reply.content == "" and gen.refusals == 1 and reply.usage["input"] == 2448


def test_concurrency_bounds_requests_in_flight():
    in_flight, peak = 0, 0

    async def handler(request):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return httpx.Response(200, json=response_json("ok"), request=request)
    gen = oc.OpenAIGenerator(MODELS, EFFORTS, "sk-test", transport=httpx.MockTransport(handler), concurrency=2, sleep=nosleep)

    async def go():
        return await asyncio.gather(*(gen.generate(Call(system="s", user_text=str(i), kind="ground")) for i in range(6)))
    assert len(run(go())) == 6 and peak == 2 and gen.calls == 6


def test_empty_key_is_refused():
    with pytest.raises(ValueError):
        oc.OpenAIGenerator(MODELS, EFFORTS, "")


def test_request_body_is_the_generators_shape_with_a_pluggable_image_part(frame):
    call = Call(system="SYS", user_text="USER", images=(frame, make_frame(100, 50)), thinking=True, kind="describe",
                history=(("assistant", "bad"), ("user", "fix it")))
    refs = iter(["file-1", None])
    body = oc.request_body(call, "gpt-5.6-terra", "low",
                           image_part=lambda im: {"type": "input_image", "file_id": next(refs), "detail": "high"} if im is frame else oc.data_url_part(im))
    content = body["input"][0]["content"]
    assert content[0] == {"type": "input_image", "file_id": "file-1", "detail": "high"}
    assert content[1]["image_url"].startswith("data:image/png;base64,") and content[2] == {"type": "input_text", "text": "SYS\n\nUSER"}
    assert body["model"] == "gpt-5.6-terra" and body["reasoning"] == {"effort": "low"} and body["max_output_tokens"] == 4096
    assert body["store"] is False and body["text"]["format"]["name"] == "annotator_describe"
    assert [m["role"] for m in body["input"]] == ["user", "assistant", "user"]
    fake = FakeOpenAI(['{"boxes": []}'])
    run(gen_for(fake).generate(Call(system="S", user_text="U", images=(frame,), kind="ground")))
    assert fake.requests[0] == oc.request_body(Call(system="S", user_text="U", images=(frame,), kind="ground"), "gpt-5.6-luna", "none")
