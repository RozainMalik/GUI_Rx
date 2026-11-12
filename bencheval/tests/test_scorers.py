import json
from pathlib import Path

from benchev.types import Sample
from benchev.scorers import score
from benchev.scorers import medspot as ms



def S(sid, gt, kind="box", **tags):
    return Sample(sid, "x", "img.png", "click", kind, gt, (100, 100), tags)


def test_pointbox_splits():
    samples = [S("a", (0, 0, .5, .5), group="web", ui_type="text"), S("b", (0, 0, .5, .5), group="web", ui_type="icon"),
               S("c", (.5, .5, 1, 1), group="mobile", ui_type="icon")]
    preds = {"a": (.2, .2), "b": (.9, .9), "c": (.7, .7)}
    m = score("pointbox", samples, preds, [["group"], ["group", "ui_type"], ["ui_type"]])
    assert m["acc"] == round(200 / 3, 2) and m["n"] == 3
    assert m["split"]["group=web"] == 50.0 and m["split"]["group=mobile/ui_type=icon"] == 100.0
    assert m["split"]["ui_type=icon"] == 50.0


def test_list_tag_counts_sample_in_every_group():
    samples = [S("a", (0, 0, .5, .5), category=["tm", "er"]), S("b", (0, 0, .5, .5), category=["er"])]
    m = score("pointbox", samples, {"a": (.2, .2), "b": (.9, .9)}, [["category"]])
    assert m["acc"] == 50.0 and m["n"] == 2
    assert m["split"] == {"category=er": 50.0, "category=tm": 100.0}
    assert m["split_n"] == {"category=er": 2, "category=tm": 1}


def test_none_point_is_miss_pointbox():
    m = score("pointbox", [S("a", (0, 0, 1, 1))], {"a": None}, [])
    assert m["acc"] == 0.0


def test_missing_prediction_counts_as_miss_and_flags():
    m = score("pointbox", [S("a", (0, 0, 1, 1)), S("b", (0, 0, 1, 1))], {"a": (.5, .5)}, [])
    assert m["acc"] == 50.0 and m["n_missing"] == 1


def test_missing_prediction_on_refusal_is_miss():
    m = score("osworld", [S("r", None, "refusal", category="x", raw_box=[0, 0, 0, 0])], {}, [])
    assert m["acc"] == 0.0 and m["n_missing"] == 1


def test_none_point_osworld_refusal_is_hit():
    samples = [S("r", None, "refusal", category="x", raw_box=[0, 0, 0, 0]),
               S("p", ((.1, .1), (.3, .1), (.3, .3), (.1, .3)), "polygon", category="x",
                 raw_box=[10, 10, 30, 10, 30, 30, 10, 30]),
               S("b", (.1, .1, .3, .3), category="y", raw_box=[10, 10, 20, 20])]
    m = score("osworld", samples, {"r": None, "p": (.2, .2), "b": (.2, .2)}, [["category"]])
    assert m["acc"] == 100.0
    # upstream-compat: refusal always miss; polygon read as xywh [10,10,30,10] -> box (.1,.1,.4,.2): (.2,.2) inside
    assert m["acc_upstream_compat"] == round(200 / 3, 2)
    assert m["acc_bbox470"] == 100.0 and m["n_bbox470"] == 1
