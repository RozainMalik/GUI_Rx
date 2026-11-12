import pytest
from annotator import contract as c


def test_object_wrapped_in_prose():
    text = 'Sure. Here is the answer:\n{"intent": "open", "candidates": []}\nHope this helps.'
    assert c.parse_object(text) == {"intent": "open", "candidates": []}


def test_object_inside_code_fence():
    text = '```json\n{"a": [1, 2, {"b": "}"}]}\n```'
    assert c.parse_object(text) == {"a": [1, 2, {"b": "}"}]}


def test_think_block_is_ignored():
    text = '<think>{"not": "this"}</think>\n{"yes": "this"}'
    assert c.parse_object(text) == {"yes": "this"}


def test_array_reply_for_grounding():
    text = '[{"bbox_2d": [10, 20, 30, 40], "label": "Save"}]'
    assert c.parse_array(text) == [{"bbox_2d": [10, 20, 30, 40], "label": "Save"}]
    assert c.parse_array("[]") == []


def test_wrong_kind_raises():
    with pytest.raises(c.ContractError):
        c.parse_array('{"a": 1}')
    with pytest.raises(c.ContractError):
        c.parse_object('[1, 2]')


def test_garbage_raises_and_has_json_reports():
    with pytest.raises(c.ContractError):
        c.parse_object("I could not find any control.")
    assert not c.has_json("nothing here")
    assert c.has_json('x {"a": 1}')


def test_brace_in_prose_before_the_object_is_skipped():
    text = 'Note {this is not json} then {"ok": true}'
    assert c.parse_object(text) == {"ok": True}


def test_repair_message_carries_the_error():
    msg = c.repair_message(ValueError("bbox is degenerate"))
    assert "bbox is degenerate" in msg
    assert "exactly one JSON" in msg


def test_parse_array_accepts_a_boxes_object_root():
    assert c.parse_array('{"boxes": [{"bbox_2d": [10, 20, 30, 40], "label": "Save"}]}') == [{"bbox_2d": [10, 20, 30, 40], "label": "Save"}]
    assert c.parse_array('{"boxes": []}') == []
    assert c.parse_array('prose [1, 2] more') == [1, 2]            # the bare array still wins
    with pytest.raises(c.ContractError):
        c.parse_array('{"bbox_2d": "none"}')
