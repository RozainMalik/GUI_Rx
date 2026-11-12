from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Sample:
    sid: str
    bench: str
    image_path: str
    instruction: str
    gt_kind: str            # box | polygon | refusal
    gt: tuple | None        # normalised: (x1, y1, x2, y2) | ((x, y), ...) | None
    size: tuple             # native (W, H) from the annotation; the runner re-reads the real size from the file
    tags: dict = field(default_factory=dict, hash=False, compare=False)
