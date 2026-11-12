import pytest

from benchev.adapters.recipes import LAYOUTS, RECIPES, SPACES, parse, render, to_norm


def test_every_recipe_has_source_known_space_and_layout():
    for r in RECIPES.values():
        assert r.source and r.space in SPACES and r.layout in LAYOUTS, r.name
        assert "{instruction}" in r.user, r.name


@pytest.mark.parametrize("parser,text,pt", [
    ("brackets", "[512,384]", (512.0, 384.0)),
    ("brackets", "[100, 200, 300, 400]", (200.0, 300.0)),
    ("mai", '<tool_call>\n{"name": "grounding", "arguments": {"action": "click", "coordinate": [512,384]}}', (512.0, 384.0)),
    ("ui_tars", "Action: click(point='<point>512 384</point>')", (512.0, 384.0)),
    ("ui_tars", "Action: click(start_box='(512,384)')", (512.0, 384.0)),
    ("qwen_tool", '<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [512, 384]}}\n</tool_call>', (512.0, 384.0)),
    ("gui_owl15", '<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [512, 384]}}\n</tool_call>', (512.0, 384.0)),
    ("paren", "(512, 384)", (512.0, 384.0)),
    ("paren_box", "(512, 384)", (512.0, 384.0)),
    ("paren_box", "(100,200,300,400)", (200.0, 300.0)),
    ("paren_box", "(100, 200),(300, 400)", (200.0, 300.0)),
    ("paren_box", "(512, -4)", (512.0, -4.0)),
    ("os_atlas", "<|object_ref_start|>language switch<|object_ref_end|><|box_start|>(576,12),(592,42)<|box_end|>", (584.0, 27.0)),
    ("os_atlas", "language switch(576,12),(592,42)", (584.0, 27.0)),
    ("aguvis", "pyautogui.click(x=0.25, y=0.75)", (0.25, 0.75)),
    ("deepseek", "<|ref|>OK<|/ref|><|det|>[[100,200,300,400]]<|/det|>", (200.0, 300.0)),
    ("cogagent", "Action: Click OK.\nGrounded Operation: CLICK(box=[[100,200,300,400]], element_info='OK')", (200.0, 300.0)),
    ("seeclick", "(0.25,0.75)", (0.25, 0.75)),
    ("sspro_tool", '<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [512, 384]}}\n</tool_call>', (512.0, 384.0)),
    ("sspro_tool", '<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [100, 200, 300, 400]}}', (200.0, 300.0)),
])
def test_parse_examples(parser, text, pt):
    assert parse(parser, text) == pt


def test_parse_box_returns_centre():
    assert parse("brackets", "[10, 20, 30, 60]") == (20.0, 40.0)
    assert parse("seeclick", "<box>(100,200),(300,400)</box>") == (0.2, 0.3)


def test_paren_box_unparsable_is_none():
    assert parse("paren_box", "Dark mode") is None


def test_qwen2vl_k1000_keeps_the_legacy_prompt():
    a, b = RECIPES["qwen2vl_legacy"], RECIPES["qwen2vl_k1000"]
    assert (a.system, a.user) == (b.system, b.user) and (b.parser, b.space) == ("paren_box", "k1000")


def test_qwen3vl_legacy_matches_the_legacy_script():
    r = RECIPES["qwen3vl_legacy"]
    sys_, user = render(r, "Open settings", (1, 1), (1, 1))
    assert sys_ is None and user == ("Open settings\nLook at the image and click the correct UI element. "
                                     "Respond ONLY with the click location in the format (x,y).")
    assert (r.parser, r.space, r.layout, r.max_new_tokens) == ("paren", "k1000", "image_text", 64)
    assert to_norm(parse(r.parser, "<think>a</think>(250, 500)"), r.space, (1, 1), (1, 1)) == (0.25, 0.5)


def test_venus_infeasible_is_none():
    assert parse("brackets", "[-1,-1]") is None


def test_gui_owl15_takes_last_match_like_upstream():
    assert parse("gui_owl15", "first (1, 2) then (30, 40)") == (30.0, 40.0)


def test_mai_regex_is_upstreams_no_space_form():
    assert parse("mai", "[512, 384]") is None      # upstream r'\[(\d+),(\d+)\]' does not allow the space


@pytest.mark.parametrize("parser", sorted({r.parser for r in RECIPES.values()}))
def test_parse_garbage_is_none(parser):
    assert parse(parser, "") is None and parse(parser, "I cannot help with that.") is None


def test_to_norm_abs_input_uses_resized_size():
    assert to_norm((500, 250), "abs_input", input_wh=(1000, 500), orig_wh=(2000, 1000)) == (0.5, 0.5)


def test_to_norm_scales():
    assert to_norm((500, 250), "k1000", (1, 1), (9, 9)) == (0.5, 0.25)
    assert to_norm((999, 0), "k999", (1, 1), (9, 9)) == (1.0, 0.0)
    assert to_norm((100, 50), "abs_orig", (1, 1), (200, 100)) == (0.5, 0.5)
    assert to_norm((0.5, 0.25), "norm1", (1, 1), (9, 9)) == (0.5, 0.25)


def test_legacy_auto_matches_the_paper_scripts():
    # evaluate_llama.py: <=1 -> fraction, <=100 -> percent, else pixels of the original image
    assert to_norm((0.5, 0.5), "legacy_auto", (1, 1), (200, 100)) == (0.5, 0.5)
    assert to_norm((50, 25), "legacy_auto", (1, 1), (200, 100)) == (0.5, 0.25)
    assert to_norm((150, 50), "legacy_auto", (1, 1), (200, 100)) == (0.75, 0.5)


def test_out_of_range_not_clamped():
    assert to_norm((1012, -5), "k1000", (1, 1), (9, 9)) == (1.012, -0.005)


def test_render_fills_placeholders_and_strips_period():
    sys_, user = render(RECIPES["qwen25vl"], "click OK", input_wh=(1288, 728), orig_wh=(1920, 1080))
    assert "The screen's resolution is 1288x728." in sys_ and user == "click OK"
    _, user = render(RECIPES["ui_venus"], "Open the menu.", (1, 1), (1, 1))
    assert "Open the menu. \n" in user and "menu.." not in user
    _, user = render(RECIPES["qwen2vl_legacy"], "x", (1, 1), (1920, 1080))
    assert "The image is 1920x1080 pixels." in user


def test_json_braces_survive_render():
    sys_, _ = render(RECIPES["kv_ground"], "x", (1, 1), (1, 1))
    assert '{"type": "function"' in sys_ and "{instruction}" not in sys_


def test_gui_owl7b_prompt_states_the_resized_size_it_is_scored_in():
    # GUI-Owl-7B (Qwen2.5-VL) answers in pixels of the screen size its prompt states (gui_owl.py)
    r = RECIPES["gui_owl"]
    sys_, _ = render(r, "x", input_wh=(2548, 1876), orig_wh=(2560, 1880))
    assert r.space == "abs_input" and "The screen's resolution is 2548x1876." in sys_ and "1000x1000" not in sys_


def test_render_non_string_instruction_as_its_repr():
    # medspot Weasis_Annotation.json#18#3 has {'text': [...]} as its instruction; the chat templates
    # print it as its repr, so every model gets that same text
    ins = {"text": ["Select the DICOM file", "Click on the Open button"]}
    assert render(RECIPES["uground"], ins, (1, 1), (1, 1))[1].count(repr(ins)) == 1


def test_sspro_tool_reads_the_prefilled_answer_cut_at_the_last_brace():
    # qwen2_5vl.py: response = guide_text + answer, cut after the last '}', then json.loads the tool call
    r = RECIPES["qwen25vl_sspro"]
    assert parse("sspro_tool", r.prefill + "640, 360]}}\n</tool_call><|im_end|> trailing") == (640.0, 360.0)
    assert parse("sspro_tool", r.prefill + "640, 360, 1]}}") is None      # neither 2 nor 4 numbers


def test_sspro_qwen_recipes_match_the_leaderboard_scripts():
    r3, r25 = RECIPES["qwen3vl_sspro"], RECIPES["qwen25vl_sspro"]
    assert (r3.space, r3.prefill, r3.parser) == ("k1000", "", "sspro_tool")
    assert r3.system.startswith("You are a helpful assistant. The user will give you an instruction, and you MUST left click")
    assert '"enum": ["left_click"]' in r3.system and "resolution is 1000x1000.\n" in r3.system   # escapes decoded
    assert (r25.space, r25.parser) == ("abs_input", "sspro_tool")
    assert r25.prefill == '<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": ['
    assert r25.system.startswith("You are a helpful assistant.\n\n\n# Tools")
    sys_, user = render(r25, "click OK", input_wh=(1288, 728), orig_wh=(1920, 1080))
    assert "The screen's resolution is 1288x728.\n" in sys_ and "{width}" not in sys_ and user == "click OK"
    assert all("ScreenSpot-Pro-GUI-Grounding" in r.source for r in (r3, r25))
