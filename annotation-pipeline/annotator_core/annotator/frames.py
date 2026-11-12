"""Which files are frames, in what order, and which apps are benchmark apps."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

# Matched on the alphanumeric prefix of the folder name.
BENCHMARK_APPS = ("3D_Slicer", "ITK_Snap", "MITK_WorkBench", "MicroDICOM", "RadiAnt", "Weasis")
_INDEX = re.compile(r"(\d+)$")
BATCH_SEED = 20260923


@dataclass(frozen=True)
class Frame:
    app: str
    name: str
    index: int
    path: Path


def frame_index(name: str) -> int:
    m = _INDEX.search(Path(name).stem)
    return int(m.group(1)) if m else -1


def normalise_app(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def is_benchmark_app(app: str) -> bool:
    n = normalise_app(app)
    return any(n.startswith(normalise_app(b)) for b in BENCHMARK_APPS)


def list_frames(root, app) -> list[Frame]:
    """Every *.png in the app folder, ordered by trailing index then name. File names are
    not assumed to repeat the folder name (JiveX DICOM Viewer/JiveX_DICOM_Viewer_001.png).
    Records are keyed by the trailing index, so two files sharing one (or a file with no
    trailing integer at all, index -1) would silently collide on one record - refused instead."""
    folder = Path(root) / app
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    frames = [Frame(app, p.name, frame_index(p.name), p) for p in files]
    frames = sorted(frames, key=lambda fr: (fr.index, fr.name))
    seen = {}
    for fr in frames:
        if fr.index == -1:
            raise ValueError(f"{app}: {fr.name} has no trailing frame index")
        if fr.index in seen:
            raise ValueError(f"{app}: duplicate frame index {fr.index}: {seen[fr.index]}, {fr.name}")
        seen[fr.index] = fr.name
    return frames


def list_apps(root) -> list[str]:
    root = Path(root)
    return sorted(d.name for d in root.iterdir() if d.is_dir() and any(p.suffix.lower() == ".png" for p in d.iterdir()))


def apps_by_size(root) -> list[str]:
    apps = list_apps(root)
    return sorted(apps, key=lambda a: (len(list_frames(root, a)), a))


def spread(frames: list, n: int) -> list:
    """n frames spread evenly across the list (the pilot's --limit), or all when n is falsy or n >= len."""
    if not n or n >= len(frames):
        return list(frames)
    step = len(frames) / n
    return [frames[int(i * step)] for i in range(n)]


def batch_frame(root, app, batch: int, seed: int = BATCH_SEED) -> Frame | None:
    """Batch k's frame of an app: the k-th of the app's frames in a fixed order shuffled by
    (seed, app). Batches never repeat a frame; None once the app has fewer than k frames."""
    if batch < 1:
        raise ValueError(f"batch numbers start at 1, got {batch}")
    frames = list_frames(root, app)
    order = random.Random(f"{seed}:{app}").sample(frames, len(frames))
    return order[batch - 1] if batch <= len(order) else None
