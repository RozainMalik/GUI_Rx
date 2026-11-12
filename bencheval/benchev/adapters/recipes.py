"""Per-model grounding recipes for the MedSPOT comparison models: prompt, message layout, answer parser, coordinate space.

Each prompt is copied verbatim from the model's official grounding code (long ones live in prompts/*.txt, extracted by
script from the source named in `source`). Where a model has no published grounding prompt, the MedSPOT evaluation
script is the source (general VLMs without a published reference). Placeholders are filled with str.replace, never
str.format, because several prompts contain JSON braces: {instruction}, {width}/{height} (the image size the model is
fed, after the processor's resize) and {orig_width}/{orig_height} (the image the adapter was given).

Parsers return the model's own numbers (boxes -> centre) or None; to_norm maps them to [0, 1] of the given image
without clamping, so an out-of-range answer stays a miss.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

PROMPTS = Path(__file__).parent / "prompts"
SPACES = {"norm1", "k1000", "k999", "abs_input", "abs_orig", "legacy_auto"}
LAYOUTS = {"image_text", "text_image", "text_then_image_turn"}


def _p(name):
    return (PROMPTS / f"{name}.txt").read_text(encoding="utf-8")


@dataclass(frozen=True)
class Recipe:
    name: str
    system: str | None
    user: str
    parser: str
    space: str
    source: str
    layout: str = "image_text"          # image_text | text_image | text_then_image_turn (text, then the image as its own user turn)
    chat_template: str | None = None    # replaces the checkpoint's template
    assistant_prefix: str | None = None  # literal assistant header used instead of the template's generation prompt
    prefill: str = ""                   # appended after the generation prompt and put back in front of the answer
    stop: str | None = None             # cut the answer at this string
    strip_period: bool = False          # drop one trailing '.' from the instruction
    max_new_tokens: int = 256           # the source's generation budget where it sets one


def render(r: Recipe, instruction: str, input_wh, orig_wh):
    """(system text or None, user text) with the placeholders filled."""
    if not isinstance(instruction, str):
        instruction = repr(instruction)   # medspot has one dict instruction; the chat templates printed it as its repr
    if r.strip_period and instruction.endswith("."):
        instruction = instruction[:-1]
    fill = {"{width}": input_wh[0], "{height}": input_wh[1], "{orig_width}": orig_wh[0], "{orig_height}": orig_wh[1]}

    def sub(t):
        for k, v in fill.items():
            t = t.replace(k, str(v))
        return t.replace("{instruction}", instruction)
    return (None if r.system is None else sub(r.system)), sub(r.user)


# ---------------------------------------------------------------- parsers (numbers in the model's own units)
_N = r"(-?\d+(?:\.\d+)?)"


def _centre(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2


def _f(*xs):
    return tuple(float(x) for x in xs)


def _brackets(t):
    """UI-Venus extract_coordinates_qwen35: [x1,y1,x2,y2] | [x1,y1],[x2,y2] | [x,y]; [-1,-1] = infeasible."""
    t = t.strip()
    for pat in (r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]",
                r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*,\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]"):
        m = re.search(pat, t)
        if m:
            v = _f(*m.groups())
            return None if v[:2] == (-1, -1) else _centre(*v)
    m = re.search(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", t)
    if m:
        v = _f(*m.groups())
        return None if v == (-1, -1) else v
    nums = _f(*re.findall(r"-?\d+\.?\d*", t))
    if len(nums) == 2:
        return None if nums == (-1, -1) else nums
    if len(nums) >= 4:
        return None if nums[:2] == (-1, -1) else _centre(*nums[:4])
    return None


def _mai(t):
    """MAI-UI parse_coordinates: first r'\\[(\\d+),(\\d+)\\]' after cutting at the last '}'."""
    i = t.rfind("}")
    t = t[:i + 1] if i != -1 else t
    m = re.search(r"\[(\d+),(\d+)\]", t)
    return _f(*m.groups()) if m else None


def _ui_tars(t):
    for pat in (rf"<point>\s*{_N}\s+{_N}\s*</point>", rf"point='\s*{_N}\s+{_N}\s*'", rf"\(\s*{_N}\s*,\s*{_N}\s*\)"):
        m = re.search(pat, t)
        if m:
            return _f(*m.groups())
    return None


def _qwen_tool(t):
    """The <tool_call> JSON's arguments.coordinate (kv-ground extract_coordinates); regex on "coordinate" if the JSON is broken."""
    if "<tool_call>" in t and "</tool_call>" in t:
        try:
            c = json.loads(t.split("<tool_call>")[1].split("</tool_call>")[0].strip())["arguments"]["coordinate"]
            return _f(c[0], c[1])
        except (ValueError, KeyError, TypeError, IndexError):
            pass
    m = re.search(rf'"coordinate"\s*:\s*\[\s*{_N}\s*,\s*{_N}', t)
    return _f(*m.groups()) if m else None


def _sspro_tool(t):
    """ScreenSpot-Pro leaderboard qwen2_5vl.py / qwen3vl.py: cut after the last '}', json.loads the last <tool_call>;
    coordinate of 2 numbers = point, of 4 = box centre, anything else (or broken JSON) = None."""
    i = t.rfind("}")
    if i == -1:
        return None
    try:
        c = json.loads(t[:i + 1].split("<tool_call>")[-1].strip())["arguments"]["coordinate"]
    except (ValueError, KeyError, TypeError):
        return None
    if isinstance(c, list) and len(c) == 2:
        return _f(*c)
    if isinstance(c, list) and len(c) == 4:
        return _centre(*_f(*c))
    return None


def _gui_owl15(t):
    """GUI-Owl-1.5 eval_grounding_benchmarks.py: all r'\\((\\d+),\\s*(\\d+)\\)', else r'\\[(\\d+), (\\d+)\\]'; the LAST one."""
    ms = re.findall(r"\((\d+),\s*(\d+)\)", t) or re.findall(r"\[(\d+), (\d+)\]", t)
    return _f(*ms[-1]) if ms else None


def _paren(t):
    """MedSPOT paper scripts: first (x, y) or x=.., y=.. after dropping <think> blocks and tags."""
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    t = re.sub(r"<.*?>", "", t, flags=re.S)
    for pat in (r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", r"x\s*=\s*([\d.]+)\s*,\s*y\s*=\s*([\d.]+)"):
        m = re.search(pat, t, re.I)
        if m:
            try:
                return _f(*m.groups())
            except ValueError:
                return None
    return None


def _paren_box(t):
    """_paren that also takes boxes: (x1,y1,x2,y2) or (x1,y1),(x2,y2) -> centre, else the first (x, y).
    For Qwen2-VL, which answers the MedSPOT pixel prompt in its native 0-1000 space and often with a box."""
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    t = re.sub(r"<.*?>", "", t, flags=re.S)
    for pat in (rf"\(\s*{_N}\s*,\s*{_N}\s*,\s*{_N}\s*,\s*{_N}\s*\)",
                rf"\(\s*{_N}\s*,\s*{_N}\s*\)\s*,\s*\(\s*{_N}\s*,\s*{_N}\s*\)"):
        m = re.search(pat, t)
        if m:
            return _centre(*_f(*m.groups()))
    m = re.search(rf"\(\s*{_N}\s*,\s*{_N}\s*\)", t)
    return _f(*m.groups()) if m else None


def _os_atlas(t):
    m = re.search(rf"\(\s*{_N}\s*,\s*{_N}\s*\)\s*,\s*\(\s*{_N}\s*,\s*{_N}\s*\)", t)
    if m:
        return _centre(*_f(*m.groups()))
    m = re.search(rf"\[\[\s*{_N}\s*,\s*{_N}\s*,\s*{_N}\s*,\s*{_N}\s*\]\]", t)
    if m:
        return _centre(*_f(*m.groups()))
    m = re.search(rf"\[\[\s*{_N}\s*,\s*{_N}\s*\]\]", t) or re.search(rf"\(\s*{_N}\s*,\s*{_N}\s*\)", t)
    return _f(*m.groups()) if m else None


def _aguvis(t):
    for pat in (rf"pyautogui\.click\s*\(\s*x\s*=\s*{_N}\s*,\s*y\s*=\s*{_N}", rf"pyautogui\.click\s*\(\s*{_N}\s*,\s*{_N}"):
        m = re.search(pat, t)
        if m:
            return _f(*m.groups())
    return None


def _deepseek(t):
    m = re.search(rf"<\|det\|>\s*\[\[\s*{_N}\s*,\s*{_N}\s*,\s*{_N}\s*,\s*{_N}\s*\]", t)
    return _centre(*_f(*m.groups())) if m else None


def _cogagent(t):
    m = re.search(rf"box\s*=\s*\[\[\s*{_N}\s*,\s*{_N}\s*,\s*{_N}\s*,\s*{_N}\s*\]\]", t)
    if m:
        return _centre(*_f(*m.groups()))
    m = re.search(rf"point\s*=\s*\[\[\s*{_N}\s*,\s*{_N}\s*\]\]", t)
    return _f(*m.groups()) if m else None


def _seeclick(t):
    """SeeClick: a (x,y) point in [0, 1]; a Qwen-VL <box>(x1,y1),(x2,y2)</box> in 0-1000 -> centre / 1000."""
    if "box" in t:
        m = re.search(rf"\(\s*{_N}\s*,\s*{_N}\s*\)\s*,\s*\(\s*{_N}\s*,\s*{_N}\s*\)", t)
        return tuple(v / 1000 for v in _centre(*_f(*m.groups()))) if m else None
    nums = re.findall(r"\d+\.\d+|\d+", t)
    return _f(*nums[:2]) if len(nums) == 2 else None


PARSERS = {"brackets": _brackets, "mai": _mai, "ui_tars": _ui_tars, "qwen_tool": _qwen_tool, "gui_owl15": _gui_owl15,
           "paren": _paren, "os_atlas": _os_atlas, "aguvis": _aguvis, "deepseek": _deepseek, "cogagent": _cogagent,
           "seeclick": _seeclick, "sspro_tool": _sspro_tool, "paren_box": _paren_box}


def parse(parser: str, text: str):
    return PARSERS[parser](text or "")


def to_norm(pt, space: str, input_wh, orig_wh):
    x, y = pt
    if space == "norm1":
        return x, y
    if space == "k1000":
        return x / 1000, y / 1000
    if space == "k999":
        return x / 999, y / 999
    if space == "abs_input":
        return x / input_wh[0], y / input_wh[1]
    if space == "abs_orig":
        return x / orig_wh[0], y / orig_wh[1]
    if space == "legacy_auto":      # evaluate_llama.py
        if 0 <= x <= 1 and 0 <= y <= 1:
            return x, y
        if 0 <= x <= 100 and 0 <= y <= 100:
            return x / 100, y / 100
        return x / orig_wh[0], y / orig_wh[1]
    raise ValueError(space)


# ---------------------------------------------------------------- recipes
_GH = "https://github.com"
_LEGACY = "MedSPOT evaluation scripts:"
VENUS_USER = ("Output the center point of the position corresponding to the following instruction: \n"
              "{instruction}. \n\n"
              "The output should just be the coordinates of a point, in the format [x,y]. "
              "Additionally, if the task is infeasible (e.g., the task is not related to the image), "
              "the output should be [-1,-1].")
UGROUND_USER = """
  Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.

  - Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
  - If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
  - Your answer should be a single string (x, y) corresponding to the point of the interest.

  Description: {instruction}

  Answer:"""
AGUVIS_SYSTEM = ("You are a GUI agent. You are given a task and a screenshot of the screen. "
                 "You need to perform a series of pyautogui actions to complete the task.")
AGUVIS_USER = ("Please generate the next move according to the ui screenshot, instruction and previous actions.\n\n"
               "Instruction: {instruction}\n\nPrevious actions:\nNone\n")
AGUVIS_TEMPLATE = (
    "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}"
    "{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
    "{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}"
    "<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}"
    "{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}"
    "<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}"
    "<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
LEGACY_COORD_USER = ("{instruction}\n\n"
                     "Return ONLY a single coordinate pair.\n"
                     "Output format must be exactly:\n"
                     "(x,y)\n\n"
                     "Do NOT explain.\n"
                     "Do NOT describe.\n"
                     "Do NOT output text.\n"
                     "If unsure, guess.\n"
                     "Your entire response must be only two numbers inside parentheses.")
MAI_PREFILL = '<tool_call>\n{"name": "grounding", "arguments": {"action": "click", "coordinate": ['
QWEN_LC_PREFILL = '<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": ['

RECIPES = {r.name: r for r in [
    Recipe("ui_venus", None, VENUS_USER, "brackets", "k1000", strip_period=True,
           source=f"{_GH}/inclusionAI/UI-Venus/blob/1ffbaf653fb3d5d08e9ea14b013865496d68408e/models/grounding/ui_venus2_gd.py"
                  " DEFAULT_USER_PROMPT (= UI-Venus-1.5 report arXiv 2602.09082 App. A.2)"),
    Recipe("mai_ui", _p("mai_ui_system"), "{instruction}\n", "mai", "k1000", layout="text_image", prefill=MAI_PREFILL,
           source=f"{_GH}/Tongyi-MAI/MAI-UI/blob/3deabf1d1fcf31964511929938b7c9485589b29c/MAI-UI/evaluation/grounding/"
                  "models/MAI_UI.py batch_ground_only_positive (use_guide_text=True, the eval_local.py default)"),
    Recipe("ui_tars15", None, _p("ui_tars15_grounding"), "ui_tars", "abs_input", layout="text_then_image_turn",
           source=f"{_GH}/bytedance/UI-TARS/blob/582f3a7ea5d285ee8ed9e2e84048d1ab01453c49/codes/ui_tars/prompt.py"
                  " GROUNDING_DOUBAO; layout as data/test_messages.json; coordinates per README_coordinates.md"),
    Recipe("qwen25vl", _p("qwen_cu_system_wh"), "{instruction}", "qwen_tool", "abs_input",
           source=f"{_GH}/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/cookbooks/computer_use.ipynb"
                  " (NousFnCallPrompt + utils/agent_function_call.py ComputerUse, display = resized input, Qwen2.5-VL convention)"),
    Recipe("qwen3vl", _p("qwen_cu_system_1000"), "{instruction}", "qwen_tool", "k1000",
           source=f"{_GH}/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/cookbooks/computer_use.ipynb"
                  " (NousFnCallPrompt + ComputerUse display 1000x1000)"),
    Recipe("qwen25vl_sspro", _p("qwen25vl_sspro_system"), "{instruction}", "sspro_tool", "abs_input", prefill=QWEN_LC_PREFILL,
           source=f"{_GH}/likaixin2000/ScreenSpot-Pro-GUI-Grounding/blob/588f1f25a59318d49cf2a657e8dc18529c723713/models/"
                  "qwen2_5vl.py get_qwen2_5vl_prompt_msg + guide_text (HF path; screen = resized input)"),
    Recipe("qwen3vl_sspro", _p("qwen3vl_sspro_system"), "{instruction}", "sspro_tool", "k1000",
           source=f"{_GH}/likaixin2000/ScreenSpot-Pro-GUI-Grounding/blob/573b40ef0425ab384b5ffc2817faafb99c1a8b60/models/"
                  "qwen3vl.py get_qwen3vl_prompt_msg (HF path, left_click only, no guide text)"),
    Recipe("qwen2vl_legacy", None,
           "{instruction}\n\nThe image is {orig_width}x{orig_height} pixels.\n"
           "Look carefully and identify the exact UI element described.\n"
           "Return the pixel coordinates of the CENTER of that element.\n"
           "Format strictly: (x,y)\n"
           "Do NOT return the image center. Do NOT explain.", "paren", "abs_orig",
           source=f"{_LEGACY}/evaluate_qwen2vl.py evaluate_step (MedSPOT paper prompt)"),
    # Same prompt as qwen2vl_legacy, but read the answer the way Qwen2-VL writes it: 0-1000 numbers, boxes -> centre
    # (reading it as pixels scores 0.1-2 %).
    Recipe("qwen2vl_k1000", None,
           "{instruction}\n\nThe image is {orig_width}x{orig_height} pixels.\n"
           "Look carefully and identify the exact UI element described.\n"
           "Return the pixel coordinates of the CENTER of that element.\n"
           "Format strictly: (x,y)\n"
           "Do NOT return the image center. Do NOT explain.", "paren_box", "k1000",
           source=f"{_LEGACY}/evaluate_qwen2vl.py evaluate_step (MedSPOT paper prompt); answers read as Qwen2-VL 0-1000"
                  " coordinates, boxes -> centre"),
    Recipe("qwen3vl_legacy", None,
           "{instruction}\nLook at the image and click the correct UI element. "
           "Respond ONLY with the click location in the format (x,y).", "paren", "k1000", max_new_tokens=64,
           source=f"{_LEGACY}/evaluate_qwen3vl0.py evaluate_step (QWEN_COORD_SCALE 1000)"),
    Recipe("os_atlas", None,
           'In this UI screenshot, what is the position of the element corresponding to the command "{instruction}" (with bbox)?',
           "os_atlas", "k1000", source="https://huggingface.co/OS-Copilot/OS-Atlas-Base-7B model card example"),
    Recipe("uground", None, UGROUND_USER, "paren", "k1000",
           source="https://huggingface.co/osunlp/UGround-V1-7B model card 'Visual Grounding Prompt' (output in [0,1000))"),
    Recipe("aguvis", AGUVIS_SYSTEM, AGUVIS_USER, "aguvis", "norm1", chat_template=AGUVIS_TEMPLATE,
           assistant_prefix="<|im_start|>assistant<|recipient|>os\n", stop="<|diff_marker|>", max_new_tokens=1024,
           source=f"{_GH}/xlang-ai/aguvis/blob/178c4187069d8aa028aaf00f209139824c00a76f/src/aguvis/serve/cli.py"
                  " generate_response(mode='grounding') + constants.py"),
    Recipe("gui_owl", _p("gui_owl_system_wh"), "{instruction}", "gui_owl15", "abs_input", max_new_tokens=2048,
           source=f"{_GH}/X-PLUG/MobileAgent/blob/11cea575561fb7800b5fb6b6cafa56f7a91de11f/Mobile-Agent-v3.5/"
                  "grounding_and_kb/eval_grounding_benchmarks.py only_two_action prompt, with the screen size set to the resized input"
                  " as Mobile-Agent-v3 gui_owl.py does for the Qwen2.5-VL-based GUI-Owl (pixels of that size)"),
    Recipe("gui_owl15", _p("gui_owl15_system"), "{instruction}", "gui_owl15", "k1000", max_new_tokens=2048,
           source=f"{_GH}/X-PLUG/MobileAgent/blob/11cea575561fb7800b5fb6b6cafa56f7a91de11f/Mobile-Agent-v3.5/"
                  "grounding_and_kb/eval_grounding_benchmarks.py only_two_action_system_prompt, use_qwen3vl=True"),
    Recipe("kv_ground", _p("kv_ground_system"), "{instruction}", "qwen_tool", "k1000",
           source=f"{_GH}/vocaela/kv-ground/blob/04cf5309cadb311882edf52d2c7a1fa75611583f/utils.py"
                  " QWEN3VL_GROUNDING_SYSTEM_MESSAGE_TEXT + extract_coordinates"),
    Recipe("legacy_coords", None, LEGACY_COORD_USER, "paren", "legacy_auto",
           source=f"{_LEGACY}/evaluate_llama.py evaluate_step (MedSPOT paper prompt; also used for Ministral, no legacy script)"),
    Recipe("deepseek_ref", None, "<image>\n<|ref|>{instruction}<|/ref|>.", "deepseek", "k999",
           source="https://huggingface.co/deepseek-ai/deepseek-vl2-small model card visual-grounding example (no legacy script)"),
    Recipe("cogagent", None,
           "Task: {instruction}\nHistory steps: \n(Platform: WIN)\n(Answer in Action-Operation-Sensitive format.)\n",
           "cogagent", "k1000",
           source="https://huggingface.co/zai-org/cogagent-9b-20241220 model card query concatenation (no history)"),
    Recipe("seeclick", None,
           'In this UI screenshot, what is the position of the element corresponding to the command "{instruction}" (with point)?',
           "seeclick", "norm1", source=f"{_LEGACY}/evaluate_seeclick.py (= SeeClick repo pred_2_point prompt)"),
]}
