import pytest
from annotator import validators as v

GOOD = {"functional": "Start a slideshow of the loaded images.",
        "visual": "Click the menu item labelled 'Slideshow...'.",
        "spatial": "Click the eighth row of the open File menu, just below 'Thumbnails'."}
NEIGHBOURS = {"above": "Thumbnails", "below": "Batch Conversion", "left": "", "right": ""}


def test_good_instructions_pass():
    assert v.check_instructions(GOOD, "Slideshow...", "", NEIGHBOURS) == []


@pytest.mark.parametrize("opener", ["The slideshow button starts it.", "A menu item that starts a slideshow.",
                                    "Clicking here starts a slideshow.", "Opens the slideshow dialog now.",
                                    "Located below Thumbnails in the menu."])
def test_description_openers_fail(opener):
    errs = v.check_instructions(dict(GOOD, functional=opener), "Slideshow...", "", NEIGHBOURS)
    assert any("must start with a verb" in e for e in errs)


def test_word_count_bounds():
    short = dict(GOOD, functional="Start slideshow.")
    long = dict(GOOD, functional="Start " + "very " * 30 + "slideshow.")
    assert any("words" in e for e in v.check_instructions(short, "Slideshow...", "", NEIGHBOURS))
    assert any("words" in e for e in v.check_instructions(long, "Slideshow...", "", NEIGHBOURS))


def test_visual_must_quote_label_verbatim_case_insensitive():
    ok = dict(GOOD, visual="Click the menu item labelled 'slideshow...'.")
    assert v.check_instructions(ok, "Slideshow...", "", NEIGHBOURS) == []
    bad = dict(GOOD, visual="Click the menu item for slideshows.")
    assert any("verbatim" in e for e in v.check_instructions(bad, "Slideshow...", "", NEIGHBOURS))


def test_icon_rule():
    with_icon_word = dict(GOOD, visual="Click the 'Slideshow...' item with the projector icon.")
    assert any("icon" in e for e in v.check_instructions(with_icon_word, "Slideshow...", "", NEIGHBOURS))
    assert v.check_instructions(with_icon_word, "Slideshow...", "projector", NEIGHBOURS) == []


def test_spatial_rule_accepts_ordinal_relation_or_neighbour():
    base = dict(GOOD)
    assert v.check_instructions(dict(base, spatial="Click the 3rd row of the File menu."), "Slideshow...", "", NEIGHBOURS) == []
    assert v.check_instructions(dict(base, spatial="Click the row just under the Thumbnails entry."), "Slideshow...", "", NEIGHBOURS) == []
    assert v.check_instructions(dict(base, spatial="Click the row next to nothing in particular."), "Slideshow...", "", NEIGHBOURS) == []
    assert v.check_instructions(dict(base, spatial="Click the row that reads Batch Conversion later."), "Slideshow...", "", NEIGHBOURS) == []
    bad = dict(base, spatial="Click the slideshow row in the menu.")
    assert any("spatial" in e for e in v.check_instructions(bad, "Slideshow...", "", NEIGHBOURS))


@pytest.mark.parametrize("word", ["cursor", "hover", "highlighted", "bounding box", "screenshot", "annotation"])
def test_forbidden_words(word):
    bad = dict(GOOD, functional=f"Start the slideshow near the {word} now.")
    assert any("must not mention" in e for e in v.check_instructions(bad, "Slideshow...", "", NEIGHBOURS))


def test_missing_variant_reported():
    errs = v.check_instructions({"functional": GOOD["functional"], "visual": GOOD["visual"]}, "Slideshow...", "", NEIGHBOURS)
    assert "spatial: missing" in errs


def test_readback_normaliser_and_match():
    assert v.normalise_label("Slideshow…") == v.normalise_label("Slideshow...") == "slideshow"
    assert v.readback_match("Slideshow...", "Slideshow…") is True
    assert v.readback_match("Save As...", "File  Save As") is True          # substring after normalising
    assert v.readback_match("Slideshow", "Slideshoe") is True               # ratio >= 0.8
    assert v.readback_match("Slideshow", "Thumbnails") is False
    assert v.readback_match("", "anything") is None
    assert v.readback_match("Open", "") is False


def test_check_box_1000():
    assert v.check_box_1000([10, 20.4, 30, 40]) == [10, 20, 30, 40]
    for bad in ([10, 20, 30], [10, 20, 10, 40], [-1, 0, 5, 5], [0, 0, 1001, 5], ["a", 0, 5, 5], None):
        with pytest.raises(v.ValidationError):
            v.check_box_1000(bad)


def test_box_sanity():
    assert v.box_sanity((100, 100, 300, 130), 1920, 1080) is None
    assert v.box_sanity((100, 100, 100, 130), 1920, 1080) == "degenerate"
    assert v.box_sanity((-1, 100, 300, 130), 1920, 1080) == "outside_frame"
    assert v.box_sanity((100, 100, 108, 130), 1920, 1080) == "too_small"
    assert v.box_sanity((0, 0, 1100, 500), 1920, 1080) == "too_large"       # 26.5 % of the frame


def test_enum_and_nonempty():
    assert v.check_enum("menu item", ("menu item", "tab"), "kind") == "menu item"
    assert v.check_enum(" Menu Item ", ("menu item", "tab"), "kind") == "menu item"    # tolerant of case/space
    with pytest.raises(v.ValidationError):
        v.check_enum("panel", ("menu item",), "kind")
    assert v.check_nonempty_str(" Save ", "name") == "Save"
    with pytest.raises(v.ValidationError):
        v.check_nonempty_str("", "name")


def test_soft_validation_error_carries_value_and_errors():
    err = v.SoftValidationError({"a": 1}, ["visual: bad"])
    assert err.value == {"a": 1} and err.errors == ["visual: bad"]
    assert isinstance(err, v.ValidationError) and "visual: bad" in str(err)


def test_check_box_bounds_follow_the_coordinate_space():
    assert v.check_box([10, 20.4, 30, 40], (1000, 1000)) == [10, 20, 30, 40]
    assert v.check_box([0, 0, 1920, 1080], (1920, 1080)) == [0, 0, 1920, 1080]
    with pytest.raises(v.ValidationError, match="0-1920 across, 0-1080 down"):
        v.check_box([0, 0, 1921, 5], (1920, 1080))
    with pytest.raises(v.ValidationError, match="0-1000 scale"):
        v.check_box([0, 0, 1001, 5])
    for bad in ([10, 20, 30], [10, 20, 10, 40], [-1, 0, 5, 5], ["a", 0, 5, 5], None):
        with pytest.raises(v.ValidationError):
            v.check_box(bad, (1920, 1080))
