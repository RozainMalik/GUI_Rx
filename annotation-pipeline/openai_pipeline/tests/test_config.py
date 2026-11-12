import pytest

import annotator
from oai import config as c
from tests.conftest import write_config


def test_loads_every_section_and_resolves_relative_paths(tmp_path):
    cfg = c.load_config(write_config(tmp_path))
    assert cfg.models["ground"] == "gpt-5.6-luna" and cfg.efforts == {"propose": "low", "describe": "low", "ground": "none", "verify": "none", "refine_ground": "none"}
    assert cfg.coord_mode == "pixels" and cfg.concurrency == 2 and cfg.seed == 7 and cfg.early_stop is False
    assert cfg.sizes == (2, 2) and cfg.gate == 0.75 and cfg.expected_apps == 2 and cfg.batch_usd == 0 and cfg.run_usd == 50
    assert cfg.work == (tmp_path / "work").resolve() and cfg.data == (tmp_path / "data").resolve()
    assert cfg.image_root == str(tmp_path / "data").replace("\\", "/") and cfg.remote == "user@host"
    assert cfg.shared_repo == c.Path(annotator.__file__).resolve().parents[1] and cfg.path == (tmp_path / "config.toml").resolve()


def test_defaults_when_optional_keys_are_absent(tmp_path):
    cfg = c.load_config(write_config(tmp_path, run='coord_mode = "thousand"', batches="", caps=""))
    assert cfg.coord_mode == "thousand" and cfg.concurrency == 4 and cfg.frame_concurrency == 4 and cfg.seed == 20260916
    assert cfg.sizes == c.DEFAULT_SIZES and cfg.gate == 0.75 and cfg.expected_apps == 30 and cfg.batch_usd == 0 and cfg.run_usd == 50
    assert cfg.dataset == "gui_grounding" and cfg.exclude_benchmark_apps is False


def test_dataset_name_and_benchmark_exclusion_are_configurable(tmp_path):
    cfg = c.load_config(write_config(tmp_path, export='dataset = "clicks_v2"', batches="sizes = [2]\nexclude_benchmark_apps = true"))
    assert cfg.dataset == "clicks_v2" and cfg.exclude_benchmark_apps is True


@pytest.mark.parametrize("section, body, message", [
    ("models", 'propose = "gpt-5.6-luna"', "[models].describe missing"),
    ("effort", 'propose = "max"\ndescribe = "lots"\nground = "none"\nverify = "none"\nrefine_ground = "none"', "[effort].describe must be one of"),
    ("run", 'coord_mode = "percent"', "[run].coord_mode must be one of"),
    ("batches", "sizes = [25, 0]", "[batches].sizes must be positive integers"),
    ("batches", "gate = 1.5", "[batches].gate must be between 0 and 1"),
    ("caps", "run_usd = -1", "[caps].run_usd must be >= 0"),
    ("export", 'dataset = "MedSPOT set"', "[export].dataset must be a lowercase slug"),
])
def test_refuses_bad_values(tmp_path, section, body, message):
    with pytest.raises(c.ConfigError, match=message.replace("[", r"\[").replace("]", r"\]")):
        c.load_config(write_config(tmp_path, **{section: body}))


def test_refuses_an_unknown_model(tmp_path):
    models = 'propose = "gpt-nope"\n' + "\n".join(f'{k} = "gpt-5.6-luna"' for k in ("describe", "ground", "verify", "refine_ground"))
    with pytest.raises(c.ConfigError, match=r"\[models\]\.propose"):
        c.load_config(write_config(tmp_path, models=models))


def test_config_hash_tracks_models_efforts_and_mode(tmp_path):
    a = c.config_hash(c.load_config(write_config(tmp_path)))
    b = c.config_hash(c.load_config(write_config(tmp_path, effort='propose = "high"\ndescribe = "low"\nground = "none"\nverify = "none"\nrefine_ground = "none"')))
    same = c.config_hash(c.load_config(write_config(tmp_path, caps="batch_usd = 9\nrun_usd = 9")))
    assert len(a) == 16 and a != b and a == same


def test_batch_transport_knobs_default_to_sync_and_are_configurable(tmp_path):
    cfg = c.load_config(write_config(tmp_path))
    assert cfg.transport == "sync" and cfg.poll_seconds == 60 and cfg.file_expiry_days == 30 and cfg.balanced is False
    cfg = c.load_config(write_config(tmp_path, run='coord_mode = "pixels"\ntransport = "batch"\npoll_seconds = 5\nfile_expiry_days = 7',
                                     batches="sizes = [2]\nbalanced = true"))
    assert cfg.transport == "batch" and cfg.poll_seconds == 5 and cfg.file_expiry_days == 7 and cfg.balanced is True
    assert c.config_hash(cfg) == c.config_hash(c.load_config(write_config(tmp_path)))     # transport never changes the answers


@pytest.mark.parametrize("body, message", [
    ('transport = "carrier-pigeon"', r"\[run\]\.transport must be one of"),
    ("poll_seconds = 0", r"\[run\]\.poll_seconds must be >= 1"),
    ("file_expiry_days = 31", r"\[run\]\.file_expiry_days must be between 1 and 30"),
])
def test_refuses_bad_transport_knobs(tmp_path, body, message):
    with pytest.raises(c.ConfigError, match=message):
        c.load_config(write_config(tmp_path, run='coord_mode = "pixels"\n' + body))


def test_part_mb_defaults_to_48_and_is_bounded(tmp_path):
    assert c.load_config(write_config(tmp_path)).part_mb == 48
    assert c.load_config(write_config(tmp_path, run='coord_mode = "pixels"\npart_mb = 20')).part_mb == 20
    with pytest.raises(c.ConfigError, match=r"\[run\]\.part_mb must be between 1 and 190"):
        c.load_config(write_config(tmp_path, run='coord_mode = "pixels"\npart_mb = 200'))
