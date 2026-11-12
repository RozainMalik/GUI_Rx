import pytest

from oai import schemas as s
from tests.conftest import BAD_REPLIES, GOOD_REPLIES


def _walk(schema, path="$"):
    """Every object node of a strict schema: additionalProperties false, required == every property."""
    t = schema.get("type")
    if t == "object" or (isinstance(t, list) and "object" in t):
        assert schema.get("additionalProperties") is False, path
        assert sorted(schema.get("required", [])) == sorted(schema["properties"]), path
        for k, v in schema["properties"].items():
            _walk(v, f"{path}.{k}")
    if "items" in schema:
        _walk(schema["items"], path + "[]")


@pytest.mark.parametrize("kind", ["propose", "describe", "ground", "refine_ground", "verify"])
def test_every_schema_is_strict(kind):
    _walk(s.schema_for(kind))
    fmt = s.text_format(kind)
    assert fmt["type"] == "json_schema" and fmt["strict"] is True and fmt["name"] == f"annotator_{kind}" and fmt["schema"] is s.schema_for(kind)


def test_probe_and_unknown_kinds_have_no_schema():
    assert s.schema_for("probe") is None and s.text_format("") is None


@pytest.mark.parametrize("kind", list(GOOD_REPLIES))
def test_good_replies_conform(kind):
    assert s.conforms(GOOD_REPLIES[kind], s.schema_for(kind)) == []


@pytest.mark.parametrize("kind, bad", [(k, b) for k, bs in BAD_REPLIES.items() for b in bs])
def test_bad_replies_do_not_conform(kind, bad):
    assert s.conforms(bad, s.schema_for(kind))


def test_enums_come_from_the_shared_package():
    from annotator.prompts import ELEMENT_KINDS, REGION_KINDS, STATES
    from annotator.stages.verify import VERDICTS
    props = s.schema_for("propose")["properties"]
    assert props["focus_region"]["properties"]["kind"]["enum"] == list(REGION_KINDS)
    cand = props["candidates"]["items"]["properties"]
    assert cand["element_kind"]["enum"] == list(ELEMENT_KINDS) and cand["state"]["enum"] == list(STATES)
    assert s.schema_for("verify")["properties"]["verdict"]["enum"] == list(VERDICTS)
    assert s.schema_for("verify")["properties"]["corrected_bbox"]["type"] == ["array", "null"]
