from pathlib import Path

import pytest

from annotator import frames as f


def make_root(tmp_path):
    for app, names in {"IrfanView": ["IrfanView_002.png", "IrfanView_001.png", "IrfanView_010.png", "notes.txt"],
                       "JiveX DICOM Viewer": ["JiveX_DICOM_Viewer_001.png"],
                       "3D_Slicer": ["3D_Slicer_001.png", "3D_Slicer_002.png"],
                       "Empty": ["readme.md"]}.items():
        d = tmp_path / app
        d.mkdir()
        for n in names:
            (d / n).write_bytes(b"")
    return tmp_path


def test_list_apps_skips_folders_without_pngs(tmp_path):
    root = make_root(tmp_path)
    assert f.list_apps(root) == ["3D_Slicer", "IrfanView", "JiveX DICOM Viewer"]


def test_list_frames_sorted_by_trailing_index(tmp_path):
    root = make_root(tmp_path)
    frames = f.list_frames(root, "IrfanView")
    assert [x.name for x in frames] == ["IrfanView_001.png", "IrfanView_002.png", "IrfanView_010.png"]
    assert [x.index for x in frames] == [1, 2, 10]
    assert frames[0].app == "IrfanView" and frames[0].path == Path(root) / "IrfanView" / "IrfanView_001.png"


def test_frame_name_need_not_match_folder_name(tmp_path):
    root = make_root(tmp_path)
    assert [x.index for x in f.list_frames(root, "JiveX DICOM Viewer")] == [1]


def test_duplicate_frame_index_raises(tmp_path):
    d = tmp_path / "App"
    d.mkdir()
    (d / "App_001.png").write_bytes(b"")
    (d / "App_v2_001.png").write_bytes(b"")
    with pytest.raises(ValueError, match="duplicate frame index"):
        f.list_frames(tmp_path, "App")


def test_frame_without_index_raises(tmp_path):
    d = tmp_path / "App"
    d.mkdir()
    (d / "noindex.png").write_bytes(b"")
    with pytest.raises(ValueError, match="no trailing frame index"):
        f.list_frames(tmp_path, "App")


def test_frame_index_fallback():
    assert f.frame_index("Screenshot_118.png") == 118
    assert f.frame_index("noindex.png") == -1


def test_apps_by_size_ascending(tmp_path):
    root = make_root(tmp_path)
    assert f.apps_by_size(root) == ["JiveX DICOM Viewer", "3D_Slicer", "IrfanView"]


def test_benchmark_apps_match_on_alphanumeric_prefix():
    for app in ["3D_Slicer", "ITK_Snap", "MITK_WorkBench", "MicroDICOM", "RadiAnt", "Weasis", "3D Slicer 5.6", "microdicom"]:
        assert f.is_benchmark_app(app), app
    for app in ["3DimViewer", "IrfanView", "MITO", "MRIcroGL", "Medisp_Lab"]:
        assert not f.is_benchmark_app(app), app


def test_spread_picks_evenly_spaced_frames():
    frames = [f.Frame("A", f"A_{i:03d}.png", i, Path(f"A_{i:03d}.png")) for i in range(1, 101)]
    picked = f.spread(frames, 10)
    assert len(picked) == 10 and picked[0].index == 1 and picked[-1].index == 91
    assert f.spread(frames, 500) == frames
    assert f.spread(frames, 0) == frames and f.spread(frames, None) == frames


def make_app(root, app, n):
    d = root / app
    d.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        (d / f"{app}_{i:03d}.png").write_bytes(b"")
    return root


def test_batch_frame_is_seeded_and_never_repeats(tmp_path):
    root = make_app(tmp_path, "IrfanView", 12)
    picks = [f.batch_frame(root, "IrfanView", k, 7).index for k in range(1, 13)]
    assert sorted(picks) == list(range(1, 13))                     # batches 1..12 cover every frame once
    assert picks == [f.batch_frame(root, "IrfanView", k, 7).index for k in range(1, 13)]
    assert picks != [f.batch_frame(root, "IrfanView", k, 8).index for k in range(1, 13)]
    assert picks != list(range(1, 13))                             # not just the recording order


def test_batch_frame_differs_per_app_and_runs_out(tmp_path):
    root = make_app(make_app(tmp_path, "A", 20), "B", 20)
    a = [f.batch_frame(root, "A", k, 7).index for k in range(1, 21)]
    b = [f.batch_frame(root, "B", k, 7).index for k in range(1, 21)]
    assert a != b
    assert f.batch_frame(root, "A", 21, 7) is None
    with pytest.raises(ValueError):
        f.batch_frame(root, "A", 0, 7)
