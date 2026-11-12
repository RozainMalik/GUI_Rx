import pytest
from PIL import Image

from annotator import calling as k
from annotator.calling import Call, Reply
from annotator.contract import ContractError, parse_object
from annotator.validators import SoftValidationError, ValidationError
from tests.conftest import FakeGenerator, run


def small():
    return Image.new("RGB", (8, 8), "white")


def test_call_defaults_and_max_tokens():
    c = k.Call(system="s", user_text="u")
    assert c.max_tokens == 512 and c.thinking is False and c.images == () and c.history == ()
    assert c.with_thinking(True).max_tokens == 4096


def test_to_openai_messages_shape():
    c = k.Call(system="sys", user_text="hello", images=(small(), small()), thinking=True)
    msgs = k.to_openai_messages(c)
    assert msgs[0] == {"role": "system", "content": "sys"}
    parts = msgs[1]["content"]
    assert msgs[1]["role"] == "user"
    assert [p["type"] for p in parts] == ["image_url", "image_url", "text"]     # images first, in order
    assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parts[2]["text"] == "hello"


def test_repair_history_is_appended_as_turns():
    c = k.Call(system="sys", user_text="hello").with_repair("bad raw", ValueError("no bbox"))
    msgs = k.to_openai_messages(c)
    assert msgs[2] == {"role": "assistant", "content": "bad raw"}
    assert msgs[3]["role"] == "user" and "no bbox" in msgs[3]["content"]


def interpret_obj(raw):
    obj = parse_object(raw)
    if "bbox" not in obj:
        raise ValidationError("missing bbox")
    return obj


def test_first_reply_good():
    gen = FakeGenerator(['{"bbox": [1, 2, 3, 4]}'])
    ex = run(k.call_validated(gen, k.Call("s", "u"), interpret_obj))
    assert ex.value == {"bbox": [1, 2, 3, 4]} and ex.calls == 1 and not ex.repaired and ex.error is None


def test_repair_once_then_good():
    gen = FakeGenerator(['{"nope": 1}', '{"bbox": [1, 2, 3, 4]}'])
    ex = run(k.call_validated(gen, k.Call("s", "u"), interpret_obj))
    assert ex.value == {"bbox": [1, 2, 3, 4]} and ex.calls == 2 and ex.repaired
    repair_call = gen.calls[1]
    assert repair_call.history[0] == ("assistant", '{"nope": 1}')
    assert "missing bbox" in repair_call.history[1][1]


def test_second_failure_gives_none_and_error():
    gen = FakeGenerator(['garbage', 'still garbage'])
    ex = run(k.call_validated(gen, k.Call("s", "u"), interpret_obj))
    assert ex.value is None and ex.calls == 2 and "no JSON" in ex.error


def test_thinking_fallback_when_content_has_no_json():
    gen = FakeGenerator([k.Reply(content="", reasoning="x" * 5000), '{"bbox": [1, 2, 3, 4]}'])
    ex = run(k.call_validated(gen, k.Call("s", "u", thinking=True), interpret_obj))
    assert ex.value is not None and ex.thinking_fallback and ex.calls == 2
    assert gen.calls[0].thinking is True and gen.calls[1].thinking is False
    assert ex.thinking_chars == 5000


def test_soft_failure_twice_keeps_value_with_errors():
    def interpret_soft(raw):
        obj = parse_object(raw)
        raise SoftValidationError(obj, ["visual: bad"])
    gen = FakeGenerator(['{"a": 1}', '{"a": 2}'])
    ex = run(k.call_validated(gen, k.Call("s", "u"), interpret_soft))
    assert ex.value == {"a": 2} and ex.error is None and ex.soft_errors == ["visual: bad"] and ex.repaired


def test_repair_reply_without_json_falls_back_to_thinking_off():
    gen = FakeGenerator(['{"nope": 1}', k.Reply(content="", reasoning="x" * 3000), '{"bbox": [1, 2, 3, 4]}'])
    ex = run(k.call_validated(gen, k.Call("s", "u", thinking=True), interpret_obj))
    assert ex.value == {"bbox": [1, 2, 3, 4]} and ex.calls == 3
    assert ex.thinking_fallback is True and ex.repaired is True
    third = gen.calls[2]
    assert third.thinking is False
    assert third.history[0] == ("assistant", '{"nope": 1}')
    assert "missing bbox" in third.history[1][1]


def test_transport_error_propagates():
    gen = FakeGenerator([k.TransportError("server gone")])
    with pytest.raises(k.TransportError):
        run(k.call_validated(gen, k.Call("s", "u"), interpret_obj))


def test_call_kind_and_reply_usage_defaults_and_survival():
    call = Call(system="s", user_text="u", kind="ground")
    assert Call(system="s", user_text="u").kind == "" and Reply(content="x").usage is None
    assert call.with_thinking(True).kind == "ground" and call.with_repair("raw", "err").kind == "ground"
    assert Reply(content="x", usage={"input": 1}).usage == {"input": 1}
