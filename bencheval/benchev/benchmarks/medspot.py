"""MedSPOT through vendored core.load_tasks: every bbox step of every task (216 tasks / 603 steps).
size (0, 0) = read from the image file at run time."""
from pathlib import Path

from medspot_strict.core import load_tasks as _core_load

from benchev.types import Sample


def load_tasks(root, limit_tasks=0):
    return _core_load(Path(root) / "Annotations", limit_tasks)


def load(root, limit_tasks=0):
    root = Path(root)
    out = []
    for t in load_tasks(root, limit_tasks):
        for s in t.steps:
            out.append(Sample(s.key, "medspot", str(root / s.image_path), s.instruction, "box", tuple(s.gt_bbox), (0, 0),
                              {"task_key": f"{t.source_file}#{t.task_idx}", "action_type": s.action_type}))
    return out
