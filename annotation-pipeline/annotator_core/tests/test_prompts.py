import re
from annotator import prompts as p
from annotator.geometry import CoordSpace, full_space


def test_render_replaces_only_named_fields_and_keeps_json_braces():
    out = p.render(p.PROPOSE_USER, app="IrfanView", kinds=", ".join(p.ELEMENT_KINDS))
    assert "Application: IrfanView" in out
    assert "toolbar button, menu item" in out
    assert '{"intent": "..."' in out                 # JSON skeleton untouched
    assert "{app}" not in out and "{kinds}" not in out


def test_every_template_renders_without_leftover_placeholders():
    rendered = [
        p.render(p.SYSTEM_ANNOTATOR, coords=p.COORDS_THOUSAND),
        p.render(p.DESCRIBE_USER, app="A", region_kind="menu", element_name="Save", text_on_screen="Save", icon="", coords=p.COORDS_THOUSAND),
        p.render(p.GROUND_USER, instruction="Click 'Save'.", coords=p.COORDS_THOUSAND),
        p.render(p.VERIFY_USER, visual="Click 'Save'.", functional="Save the file.", coords=p.COORDS_THOUSAND),
        p.render(p.REFINE_GROUND_USER, app="A", instruction="Save the file.",
                 previous=p.render(p.REFINE_WRONG_SENTENCE, text_in_box="Open"),
                 target=p.render(p.REFINE_LABEL_CLAUSE, label="Save", icon_clause=" and carries a floppy disk icon"), coords=p.COORDS_THOUSAND),
    ]
    for text in rendered:
        assert not re.search(r"\{[a-z_]+\}", text), text


def test_enums():
    assert p.ELEMENT_KINDS[0] == "toolbar button" and "other" in p.ELEMENT_KINDS and len(p.ELEMENT_KINDS) == 13
    assert p.STATES == ("pressed", "highlighted", "selected", "hovered", "focused", "plain")
    assert p.REGION_KINDS == ("menu", "dialog", "toolbar", "panel", "tab strip", "list", "other")


def test_system_prompts():
    assert p.SYSTEM_PLAIN == "You are a helpful assistant."
    assert p.SYSTEM_ANNOTATOR.startswith("You are an expert annotator")
    assert "6. Think as long as you need" in p.SYSTEM_ANNOTATOR


def test_prompt_hash_is_sha256_and_tracks_constants(monkeypatch):
    h1 = p.prompt_hash()
    assert re.fullmatch(r"[0-9a-f]{64}", h1)
    assert p.prompt_hash() == h1
    monkeypatch.setattr(p, "GROUND_USER", p.GROUND_USER + " ")
    assert p.prompt_hash() != h1


def test_thousand_mode_sentences_are_the_original_qwen_wording():
    c = p.coords_clause(full_space(1920, 1080))
    assert c == "on a 0-1000 scale of"
    assert ("as integers\n   on a 0-1000 scale of the width and height of THE IMAGE YOU ARE LOOKING AT.\n"
            in p.render(p.SYSTEM_ANNOTATOR, coords=c))
    assert p.render(p.GROUND_USER, instruction="x", coords=c).endswith(
        "Coordinates are integers on a 0-1000 scale of the image width and height.")
    assert "as [x1, y1, x2, y2] on a 0-1000 scale of\n   IMAGE 2; null if" in p.render(p.VERIFY_USER, visual="v", functional="f", coords=c)
    assert "on a 0-1000 scale of this image. If the control is not in this image,\noutput []." in p.render(
        p.REFINE_GROUND_USER, app="A", instruction="i", previous="", target="", coords=c)
    for t in (p.SYSTEM_ANNOTATOR, p.GROUND_USER, p.VERIFY_USER, p.REFINE_GROUND_USER):
        assert t.count("{coords}") == 1 and "0-1000" not in t


def test_pixels_mode_names_the_sent_size():
    assert p.coords_clause(CoordSpace((10, 10, 110, 60), (400, 200), "pixels")) == "in pixel coordinates (400x200 pixels) of"
    text = p.render(p.GROUND_USER, instruction="x", coords=p.coords_clause(full_space(1920, 1080, "pixels")))
    assert text.endswith("Coordinates are integers in pixel coordinates (1920x1080 pixels) of the image width and height.")
    assert "0-1000" not in text


def test_prompt_hash_covers_the_coordinate_mode():
    assert p.prompt_hash() == p.prompt_hash("thousand")
    assert p.prompt_hash("pixels") != p.prompt_hash("thousand")
