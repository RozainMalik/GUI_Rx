import json
import os
import subprocess
import sys

import pytest
from gui_aima.qwen35.inference import sspro_crop_px


def test_sspro_crop_matches_upstream_rule():
    # upstream: int(sqrt(W*H*616^2/5760000)/28)*28 -> at the 3200x1800 reference the crop is 616 px (22 tokens of 28)
    assert sspro_crop_px(3200, 1800, multiple=28) == 616
    assert sspro_crop_px(3200, 1800) == 608  # 19 Qwen3.5 tokens of 32 px
    assert sspro_crop_px(3840, 2160) % 32 == 0
    assert sspro_crop_px(10, 10) == 32  # never degenerate


@pytest.fixture(scope="module")
def pointer_ckpt(tmp_path_factory):
    """The tiny Qwen3.5 saved as a pointer checkpoint (pointer tokens added), like train.py output."""
    from transformers import AutoProcessor, AutoTokenizer
    from gui_aima.constants import ADDITIONAL_SPECIAL_TOKENS
    from gui_aima.qwen35.modeling import Qwen3_5ForConditionalGenerationWithPointer
    tiny = "tests/_artifacts/tiny_qwen35"
    if not os.path.exists(os.path.join(tiny, "config.json")):
        subprocess.run([sys.executable, "scripts/make_tiny_qwen35.py", "--out", tiny], check=True)
    out = str(tmp_path_factory.mktemp("ckpt"))
    tok = AutoTokenizer.from_pretrained(tiny)
    tok.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    proc = AutoProcessor.from_pretrained(tiny)
    proc.tokenizer = tok
    m = Qwen3_5ForConditionalGenerationWithPointer.from_pretrained(tiny)
    m.resize_token_embeddings(len(tok))
    m.save_pretrained(out)
    proc.save_pretrained(out)
    return out


def _run(script, *args):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(["src", "tests", "."]))
    subprocess.run([sys.executable, script, "--device", "cpu", *args], check=True, env=env)


@pytest.mark.parametrize("two_step", [False, True])
def test_screenspot_pro_eval_smoke(pointer_ckpt, tmp_path, two_step):
    from PIL import Image
    data = tmp_path / "sspro"
    (data / "images" / "app").mkdir(parents=True)
    (data / "annotations").mkdir()
    Image.new("RGB", (640, 360), "white").save(data / "images" / "app" / "a.png")
    rows = [
        {"img_filename": "app/a.png", "bbox": [0, 0, 640, 360], "instruction": "Click anywhere.", "id": "a0",
         "group": "Dev", "ui_type": "text"},  # whole image: must hit
        {"img_filename": "app/a.png", "bbox": [0, 0, 0, 0], "instruction": "Click nothing.", "id": "a1",
         "group": "CAD", "ui_type": "icon"},  # empty box: must miss
    ]
    json.dump(rows, open(data / "annotations" / "all.json", "w"))
    out = tmp_path / "out"
    args = ["--ckpt", pointer_ckpt, "--data", str(data), "--out", str(out), "--max_pixels", str(200 * 32 * 32),
            "--resize_to_pixels", "0"] + (["--two_step"] if two_step else [])
    _run("eval/screenspot_pro_qwen35.py", *args)
    name = "sspro_" + os.path.basename(pointer_ckpt) + ("_2step" if two_step else "")
    res = json.load(open(out / f"{name}.json"))
    assert res["n"] == 2 and res["Dev-text"] == 1.0 and res["CAD-icon"] == 0.0 and res["All-avg"] == 0.5


def test_screenspot_v2_eval_smoke(pointer_ckpt, tmp_path):
    from PIL import Image
    data = tmp_path / "ssv2"
    data.mkdir()
    Image.new("RGB", (640, 360), "white").save(data / "a.png")
    rows = [{"img_filename": "a.png", "bbox": [0, 0, 640, 360], "instruction": "Click.", "split": "web",
             "data_type": "text"}]
    json.dump(rows, open(data / "subset.json", "w"))
    _run("eval/screenspot_v2_qwen35.py", "--ckpt", pointer_ckpt, "--data", str(data), "--out", str(tmp_path / "out"),
         "--max_pixels", str(200 * 32 * 32))
    res = json.load(open(tmp_path / "out" / (os.path.basename(pointer_ckpt) + ".json")))
    assert res["all"] == 1.0 and res["n"] == 1
