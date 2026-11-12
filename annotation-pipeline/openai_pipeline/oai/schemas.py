"""One strict JSON schema per Call.kind, built from the shared enums so a change to
ELEMENT_KINDS or VERDICTS can never drift from what the API enforces. Strict mode wants every
property required and additionalProperties false; a box is an integer array (its length and
range are the shared validators' job, and a bad one goes through the normal repair)."""
from __future__ import annotations

from annotator.prompts import ELEMENT_KINDS, REGION_KINDS, STATES
from annotator.stages.describe import NEIGHBOUR_KEYS
from annotator.stages.verify import VERDICTS
from annotator.validators import VARIANTS


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _str() -> dict:
    return {"type": "string"}


def _enum(values) -> dict:
    return {"type": "string", "enum": list(values)}


BOX = {"type": "array", "items": {"type": "integer"}}
NULLABLE_BOX = {"type": ["array", "null"], "items": {"type": "integer"}}

PROPOSE = _obj({
    "intent": _str(),
    "focus_region": _obj({"bbox": BOX, "kind": _enum(REGION_KINDS), "why": _str()}),
    "candidates": {"type": "array", "items": _obj({
        "rank": {"type": "integer"}, "element_name": _str(), "element_kind": _enum(ELEMENT_KINDS),
        "text_on_screen": _str(), "icon": _str(), "state": _enum(STATES), "bbox": BOX})},
})
DESCRIBE = _obj({
    "element_present": {"type": "boolean"}, "bbox": NULLABLE_BOX, "text_on_screen": _str(), "icon": _str(),
    "neighbours": _obj({k: _str() for k in NEIGHBOUR_KEYS}), "position": _str(),
    "instructions": _obj({v: _str() for v in VARIANTS}),
})
BOXES = _obj({"boxes": {"type": "array", "items": _obj({"bbox_2d": BOX, "label": _str()})}})
VERIFY = _obj({"text_in_box": _str(), "element_in_box": _str(), "verdict": _enum(VERDICTS), "reason": _str(),
               "corrected_bbox": NULLABLE_BOX})

SCHEMAS = {"propose": PROPOSE, "describe": DESCRIBE, "ground": BOXES, "refine_ground": BOXES, "verify": VERIFY}


def schema_for(kind) -> dict | None:
    return SCHEMAS.get(kind)


def text_format(kind) -> dict | None:
    """The Responses API `text.format` block, or None for kinds answered in free text (the probe)."""
    schema = schema_for(kind)
    if schema is None:
        return None
    return {"type": "json_schema", "name": f"annotator_{kind}", "strict": True, "schema": schema}


_TYPES = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "boolean": bool, "null": type(None)}


def conforms(value, schema, path="$") -> list[str]:
    """Errors for the subset of JSON Schema these schemas use: type (incl. a nullable list),
    enum, properties/required/additionalProperties, items. For tests and a cheap sanity check;
    the API enforces the schema at generation time."""
    types = schema.get("type")
    types = types if isinstance(types, list) else [types]
    ok = False
    for t in types:
        py = _TYPES[t]
        if isinstance(value, py) and not (t in ("integer", "number") and isinstance(value, bool)):
            ok = True
    if not ok:
        return [f"{path}: expected {'|'.join(types)}, got {type(value).__name__}"]
    errors = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in {schema['enum']}")
    if isinstance(value, dict) and "properties" in schema:
        for k in schema.get("required", []):
            if k not in value:
                errors.append(f"{path}.{k}: missing")
        for k, v in value.items():
            if k in schema["properties"]:
                errors += conforms(v, schema["properties"][k], f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{k}: not allowed")
    if isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            errors += conforms(v, schema["items"], f"{path}[{i}]")
    return errors
