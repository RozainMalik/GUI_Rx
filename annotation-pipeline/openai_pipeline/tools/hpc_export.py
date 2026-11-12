"""Rewrite an export's image paths for another machine and stage it under export/hpc_<tag>/<name>/.

    python tools/hpc_export.py <name> <remote_image_root> [--tag primary]

Every string in the export's JSON that starts with the local image_root (from the manifest) gets the
remote root instead; manifest["image_root"] follows. Nothing else changes, so the staged copy verifies
with the shared hpc/verify_annotations.py on the target machine."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def rewrite(obj, local_root: str, remote_root: str):
    if isinstance(obj, str):
        return remote_root + obj[len(local_root):] if obj.startswith(local_root) else obj
    if isinstance(obj, list):
        return [rewrite(x, local_root, remote_root) for x in obj]
    if isinstance(obj, dict):
        return {k: rewrite(v, local_root, remote_root) for k, v in obj.items()}
    return obj


def stage(name: str, remote_root: str, tag: str) -> Path:
    src = HERE / "export" / name
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    local_root = manifest["image_root"].rstrip("/")
    remote_root = remote_root.rstrip("/")
    out = HERE / "export" / f"hpc_{tag}" / name
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src.glob("*.json"):
        data = json.loads(f.read_text(encoding="utf-8"))
        (out / f.name).write_text(json.dumps(rewrite(data, local_root, remote_root), indent=1, ensure_ascii=False) + "\n",
                                  encoding="utf-8", newline="\n")
        n += 1
    print(f"STAGED {out} files={n} image_root={remote_root} frames={manifest['frames']}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("remote_image_root")
    ap.add_argument("--tag", default="primary")
    args = ap.parse_args(argv)
    stage(args.name, args.remote_image_root, args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
