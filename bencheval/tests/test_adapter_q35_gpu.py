import os

import pytest

pytestmark = pytest.mark.gpu


def test_predicts_same_point_as_port_ground():
    from PIL import Image
    from gui_aima.qwen35.inference import ground
    from benchev.adapters.gui_aima_qwen35 import GuiAimaQwen35
    from benchev.config import load_model
    cfg = load_model("ours-2b-ab-clin")
    img = Image.open(os.environ["BENCHEV_TEST_IMAGE"]).convert("RGB")
    a = GuiAimaQwen35()
    a.load(cfg, "cuda")
    p, _ = a.predict(img, "close the window", 5_860_400)
    ref = ground(a.model, a.processor, img, "close the window", 5_860_400)["point"]
    assert p == tuple(ref)
