"""The smoke run must exercise the largest images (peak memory scales with visual tokens)."""
import json
import subprocess
import sys

import yaml
from PIL import Image


def _dataset(root, name, sizes):
    img = root / "images" / name
    img.mkdir(parents=True)
    rows = []
    for i, (w, h) in enumerate(sizes):
        Image.new("RGB", (w, h), "white").save(img / f"{i}.png")
        rows.append({"image": f"{i}.png", "conversations": [{"from": "human", "value": "<image> x"},
                                                           {"from": "gpt", "value": "pyautogui.click(x=0.5, y=0.5)"}]})
    (root / f"{name}.json").write_text(json.dumps(rows))
    return {"json_path": str(root / f"{name}.json"), "images_folder": str(img) + "/", "sampling_strategy": "all"}


def test_picks_the_largest_images_across_datasets(tmp_path):
    src = tmp_path / "src"
    cfg = {"datasets": [_dataset(src, "a", [(100, 100), (3000, 2000), (200, 50)]),
                        _dataset(src, "b", [(4000, 3000), (10, 10)])]}
    (src / "mix.yaml").write_text(yaml.safe_dump(cfg))
    out = tmp_path / "smoke"
    r = subprocess.run([sys.executable, "scripts/make_smoke_data.py", "--data", str(src / "mix.yaml"), "--top", "2",
                        "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = yaml.safe_load((out / "smoke.yaml").read_text())["datasets"]
    picked = {}
    for d in got:
        rows = json.load(open(d["json_path"]))
        picked[d["images_folder"]] = [row["image"] for row in rows]
        assert d["sampling_strategy"] == "all"
    assert picked == {str(src / "images" / "b") + "/": ["0.png"], str(src / "images" / "a") + "/": ["1.png"]}
    assert "4000x3000" in r.stdout                          # reports the largest size it found
