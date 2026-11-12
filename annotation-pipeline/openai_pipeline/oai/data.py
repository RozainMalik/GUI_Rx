"""init-data: pull the app folders and the benchmark set this machine lacks with
`ssh ... tar czf - | tar xzf -`, make the `data` junction, count what is present
and print DATA_OK. Idempotent: a folder that already holds PNGs is never pulled again."""
from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

from annotator.frames import list_apps, list_frames

from .config import Config

MISSING_APPS = ("3DimViewer", "3D_Slicer", "Cell_profiler", "ITK_Snap", "medDream", "Medisp_Lab", "MicroDICOM",
                "MITK_WorkBench", "Onis_full", "QuPath", "RadiAnt", "Starviewer", "Weasis")
SSH = ["ssh", "-o", "BatchMode=yes"]          # key auth only; a missing key fails fast instead of hanging on a prompt


def run_pipe(remote_cmd, local_cmd) -> int:
    """remote_cmd | local_cmd; the remote's exit code wins when both fail."""
    src = subprocess.Popen(remote_cmd, stdout=subprocess.PIPE)
    dst = subprocess.run(local_cmd, stdin=src.stdout)
    src.stdout.close()
    return src.wait() or dst.returncode


def run_cmd(cmd) -> int:
    return subprocess.run(cmd, capture_output=True).returncode


def _has_files(folder: Path, suffix) -> bool:
    return folder.is_dir() and any(p.suffix.lower() == suffix for p in folder.rglob("*"))


def _remote_tar(remote, remote_dir, name) -> list:
    """`ssh <remote> tar czf - -C <remote_dir> <name>` - the producer side of every pull."""
    return SSH + [remote, f"tar czf - -C {shlex.quote(remote_dir)} {shlex.quote(name)}"]


def _pull(remote, remote_dir, name, dest: Path, runner) -> int:
    """<remote>:<remote_dir>/<name> -> <dest>, via <dest.parent>/.<dest.name>.partial so a dropped
    connection never leaves a half folder that looks complete."""
    part = dest.parent / f".{dest.name}.partial"
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    rc = runner(_remote_tar(remote, remote_dir, name), ["tar", "xzf", "-", "-C", str(part)])
    if rc == 0 and (part / name).is_dir():
        shutil.rmtree(dest, ignore_errors=True)          # a folder with no PNGs/JSON from before this fix
        (part / name).rename(dest)
    shutil.rmtree(part, ignore_errors=True)
    return rc if rc != 0 else (0 if dest.is_dir() else 1)


def pull_tree(remote, remote_dir, name, local_dir, runner=run_pipe) -> int:
    """<remote>:~/<remote_dir>/<name> -> <local_dir>/<name>, skipped when it is already there."""
    local_dir = Path(local_dir)
    if _has_files(local_dir / name, ".png"):
        return 0
    local_dir.mkdir(parents=True, exist_ok=True)
    return _pull(remote, remote_dir, name, local_dir / name, runner)


def ensure_junction(link: Path, target: Path, runner=run_cmd) -> bool:
    """A directory junction (no admin rights needed) so records and configs see a short path."""
    link = Path(link)
    if link.exists():
        return False
    return runner(["cmd", "/c", "mklink", "/J", str(link), str(Path(target))]) == 0


def count_data(data_root, benchmark_root) -> dict:
    data_root, benchmark_root = Path(data_root), Path(benchmark_root)
    apps = list_apps(data_root) if data_root.is_dir() else []
    frames = sum(len(list_frames(data_root, a)) for a in apps)
    images = sum(1 for p in (benchmark_root / "Images").rglob("*.png")) if (benchmark_root / "Images").is_dir() else 0
    return {"apps": len(apps), "frames": frames, "benchmark_images": images}


def init_data(cfg: Config, log=print, pipe=run_pipe, cmd=run_cmd, apps=MISSING_APPS) -> dict:
    root = Path(cfg.image_root)
    failed = []
    for app in apps:
        if pull_tree(cfg.remote, cfg.remote_data, app, root, runner=pipe) != 0:
            failed.append(app)
            log(f"pull failed: {app}")
    bench = Path(cfg.benchmark)
    if not _has_files(bench / "Annotations", ".json"):
        # the remote benchmark folder lands as <bench.parent>/<its name> and is renamed when config names it differently
        remote_parent, remote_name = str(Path(cfg.remote_benchmark).parent), Path(cfg.remote_benchmark).name
        bench.parent.mkdir(parents=True, exist_ok=True)
        if _pull(cfg.remote, remote_parent, remote_name, bench, pipe) != 0:
            failed.append("benchmark")
            log("pull failed: benchmark")
    if str(Path(cfg.data).resolve()) != str(root.resolve()):
        ensure_junction(cfg.data, root, runner=cmd)
        if not Path(cfg.data).is_dir():
            failed.append("junction")
            log(f"junction failed: {cfg.data} -> {root}")
    counts = count_data(root, bench)
    tail = " ".join(f"{k}={v}" for k, v in counts.items())
    if failed:
        log(f"DATA_INCOMPLETE {tail} failed={','.join(failed)}")
    else:
        log(f"DATA_OK {tail}")
    return counts
