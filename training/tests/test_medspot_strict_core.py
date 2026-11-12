import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medspot_strict.core import (  # noqa: E402
    PredictionLog, compute_metrics, is_hit, load_tasks, make_record,
    percent_xywh_to_norm_xyxy,
)


def write_annotations(d, name, tasks):
    Path(d, name).write_text(json.dumps({"tasks": tasks}), encoding="utf-8")


def step(img, instr, actions):
    return {"image_path": img, "instruction": instr, "actions": actions}


# bbox in percent xywh -> normalised (0.10, 0.20, 0.30, 0.40)
BOX = [10, 20, 20, 20]
INSIDE = (0.2, 0.3)
OUTSIDE = (0.9, 0.9)


class HitRuleTest(unittest.TestCase):
    def test_inside_is_hit(self):
        self.assertTrue(is_hit(INSIDE, (0.1, 0.2, 0.3, 0.4)))

    def test_edges_are_inclusive(self):
        b = (0.1, 0.2, 0.3, 0.4)
        for p in [(0.1, 0.2), (0.3, 0.4), (0.1, 0.4), (0.3, 0.2)]:
            self.assertTrue(is_hit(p, b), p)

    def test_just_outside_is_miss_no_tolerance(self):
        b = (0.1, 0.2, 0.3, 0.4)
        eps = 1e-6
        for p in [(0.1 - eps, 0.3), (0.3 + eps, 0.3), (0.2, 0.2 - eps), (0.2, 0.4 + eps)]:
            self.assertFalse(is_hit(p, b), p)

    def test_out_of_image_point_not_clamped(self):
        # a box touching the right edge must not catch a point at x=1.05
        self.assertFalse(is_hit((1.05, 0.5), (0.9, 0.4, 1.0, 0.6)))

    def test_none_is_miss(self):
        self.assertFalse(is_hit(None, (0, 0, 1, 1)))

    def test_bbox_conversion(self):
        self.assertEqual(percent_xywh_to_norm_xyxy(BOX), (0.1, 0.2, 0.3, 0.4))


class LoadTasksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        write_annotations(self.d, "B_Annotation.json", [
            {"task_overview": "t0", "steps": [
                step("a.png", "click it", [{"type": "click", "bbox": BOX}]),
                step("b.png", "drag it", [{"type": "drag", "bbox": BOX}]),
                step("c.png", "hover", [{"type": "hover", "bbox": BOX}]),
                step("d.png", "type", [{"type": "type", "bbox": BOX, "text": "x"}]),
                step("e.png", "dbl", [{"type": "double_click", "bbox": BOX}]),
                step("f.png", "no bbox", []),
                step("g.png", "two clicks", [{"type": "click", "bbox": BOX},
                                             {"type": "click", "bbox": [50, 50, 10, 10]}]),
            ]},
            {"task_overview": "t1", "steps": [step("h.png", "nothing", [{"type": "scroll"}])]},
        ])
        write_annotations(self.d, "A_Annotation.json", [
            {"task_overview": "a0", "steps": [step("x.png", "c", [{"type": "click", "bbox": BOX}])]},
            {"task_overview": "a1", "steps": [step("y.png", "c", [{"type": "click", "bbox": BOX}])]},
        ])

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_bbox_action_type_is_evaluable(self):
        t0 = [t for t in load_tasks(self.d) if t.overview == "t0"][0]
        self.assertEqual([s.action_type for s in t0.steps],
                         ["click", "drag", "hover", "type", "double_click", "click"])

    def test_step_without_bbox_is_skipped_but_indices_kept(self):
        t0 = [t for t in load_tasks(self.d) if t.overview == "t0"][0]
        self.assertEqual([s.step_idx for s in t0.steps], [1, 2, 3, 4, 5, 7])

    def test_first_bbox_action_is_gt(self):
        t0 = [t for t in load_tasks(self.d) if t.overview == "t0"][0]
        self.assertEqual(t0.steps[-1].gt_bbox, (0.1, 0.2, 0.3, 0.4))

    def test_files_sorted_and_keys_unique(self):
        tasks = load_tasks(self.d)
        self.assertEqual(tasks[0].source_file, "A_Annotation.json")
        keys = [s.key for t in tasks for s in t.steps]
        self.assertEqual(len(keys), len(set(keys)))

    def test_limit_tasks_is_per_file(self):
        tasks = load_tasks(self.d, limit_tasks=1)
        self.assertEqual([t.overview for t in tasks], ["a0", "t0"])


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        write_annotations(self.tmp.name, "App_Annotation.json", [
            {"steps": [step("1.png", "a", [{"type": "click", "bbox": BOX}]),
                       step("2.png", "b", [{"type": "drag", "bbox": BOX}])]},
            {"steps": [step("3.png", "c", [{"type": "click", "bbox": BOX}]),
                       step("4.png", "d", [{"type": "click", "bbox": BOX}])]},
            {"steps": [step("5.png", "e", [])]},
        ])
        self.tasks = load_tasks(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def recs(self, points):
        steps = [s for t in self.tasks for s in t.steps]
        return {s.key: make_record(s, p, "") for s, p in zip(steps, points)}

    def test_all_steps_scored_not_prefix(self):
        # task0: hit, hit -> complete; task1: miss, hit -> incomplete
        m = compute_metrics(self.tasks, self.recs([INSIDE, INSIDE, OUTSIDE, INSIDE]))
        self.assertEqual(m["total_tasks"], 2)
        self.assertEqual(m["tasks_without_evaluable_steps"], 1)
        self.assertEqual(m["TCA (%)"], 50.0)
        self.assertEqual(m["SHR (%)"], 75.0)            # 3/4 counts the hit after the miss
        self.assertEqual(m["prefix_SHR (%)"], 50.0)     # 2/4, legacy early-stop style
        self.assertEqual(m["step1_accuracy (%)"], 50.0)
        self.assertTrue(m["complete_run"])

    def test_drag_step_miss_fails_task(self):
        m = compute_metrics(self.tasks, self.recs([INSIDE, OUTSIDE, INSIDE, INSIDE]))
        self.assertEqual(m["TCA (%)"], 50.0)
        self.assertEqual(m["SHR_by_action_type (%)"]["drag"], 0.0)

    def test_no_prediction_counts_as_miss(self):
        m = compute_metrics(self.tasks, self.recs([INSIDE, INSIDE, None, None]))
        self.assertEqual(m["no_prediction_steps"], 2)
        self.assertEqual(m["TCA (%)"], 50.0)

    def test_missing_steps_flag_incomplete_run(self):
        m = compute_metrics(self.tasks, self.recs([INSIDE, INSIDE]))
        self.assertFalse(m["complete_run"])
        self.assertEqual(m["missing_steps"], 2)

    def test_stored_hit_field_is_ignored(self):
        recs = self.recs([OUTSIDE] * 4)
        for r in recs.values():
            r["hit"] = 1
        self.assertEqual(compute_metrics(self.tasks, recs)["SHR (%)"], 0.0)


class PredictionLogTest(unittest.TestCase):
    def test_roundtrip_and_torn_last_line(self):
        with tempfile.TemporaryDirectory() as d:
            log = PredictionLog(Path(d, "sub", "p.jsonl"))
            log.append({"key": "a", "pred_point": [0.1, 0.2]})
            log.append({"key": "b", "pred_point": None})
            with open(log.path, "a", encoding="utf-8") as f:
                f.write('{"key": "c", "pred_po')          # crash mid-write
            recs = log.load()
            self.assertEqual(sorted(recs), ["a", "b"])
            log.append({"key": "c", "pred_point": [0.5, 0.5]})
            self.assertIn("c", log.load())


if __name__ == "__main__":
    unittest.main()
