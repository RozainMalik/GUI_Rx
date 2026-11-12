import json
import os
import subprocess
import sys

import pytest
import torch
from PIL import Image

TINY = "tests/_artifacts/tiny_qwen35"


@pytest.fixture(scope="session")
def tiny_dir():
    if not os.path.exists(os.path.join(TINY, "config.json")):
        subprocess.run([sys.executable, "scripts/make_tiny_qwen35.py", "--out", TINY], check=True)
    return TINY


@pytest.fixture(scope="session")
def sample_json(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    Image.new("RGB", (1366, 768), "white").save(d / "shot.png")     # not a multiple of 32
    rec = [{"image": "shot.png", "conversations": [
        {"from": "human", "value": "<image> Click the save button."},
        {"from": "gpt", "value": "pyautogui.click(x=0.5000, y=0.2500)", "recipient": "os", "end_turn": True,
         "bbox_gt": [0.45, 0.20, 0.55, 0.30]}]}]
    (d / "train.json").write_text(json.dumps(rec))
    return d


def _dataset(tiny_dir, sample_json, max_pixels=400 * 32 * 32):
    import gui_aima.train_common as train
    from gui_aima.dataset import LazySupervisedDataset
    from transformers import AutoProcessor, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tiny_dir, model_max_length=8192, padding_side="right")
    tok.add_special_tokens({"additional_special_tokens": train.ADDITIONAL_SPECIAL_TOKENS})
    proc = AutoProcessor.from_pretrained(tiny_dir, min_pixels=3136, max_pixels=max_pixels)
    proc.tokenizer = tok
    args = train.DataArguments(data_path=str(sample_json / "train.json"), image_folder=str(sample_json),
                               max_pixels=max_pixels, image_patch_size=16)
    args.processor = proc
    return LazySupervisedDataset(tokenizer=tok, processor=proc, data_path=args.data_path, data_args=args,
                                 number_of_points=1), tok, proc


def test_labels_match_visual_token_count(tiny_dir, sample_json):
    ds, tok, proc = _dataset(tiny_dir, sample_json)
    item = ds[0]
    t, h, w = item["image_grid_thw"][0].tolist()
    n_vis = t * h * w // 4
    image_id = tok.convert_tokens_to_ids("<|image_pad|>")
    assert int((item["input_ids"] == image_id).sum()) == n_vis
    lab = item["multi_patch_labels"]
    assert lab.shape[-1] == n_vis and lab.sum() > 0
    gw = w // 2
    peak = int(lab.reshape(-1).argmax())
    cx, cy = (peak % gw + 0.5) / gw, (peak // gw + 0.5) / (h // 2)
    assert 0.40 <= cx <= 0.60 and 0.15 <= cy <= 0.35          # peak near the bbox centre


def test_collator_adds_mm_token_type_ids(tiny_dir, sample_json):
    import gui_aima.train_common as train
    ds, tok, _ = _dataset(tiny_dir, sample_json)
    image_id = tok.convert_tokens_to_ids("<|image_pad|>")
    col = train.DataCollatorForSupervisedDataset(tokenizer=tok, image_token_id=image_id)
    batch = col([ds[0], ds[0]])
    assert torch.equal(batch["mm_token_type_ids"], (batch["input_ids"] == image_id).int())
