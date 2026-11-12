from benchev.config import load_protocol, load_bench, load_model, all_models, all_benches

BENCHES = {"ssv1", "ssv2", "sspro", "oswg", "oswg_refined", "uivision", "mmbench_l2", "medspot"}


def test_all_benches_have_canonical_protocol():
    assert set(all_benches()) == BENCHES
    for b in BENCHES:
        cfg = load_bench(b)
        p = load_protocol(cfg.canonical)
        assert p.bench == b and p.id.endswith("@v1")
        assert set(p.max_pixels) == {"1step", "2step"}


def test_sspro_protocol_matches_port_eval():
    p = load_protocol("sspro@v1")
    assert p.max_pixels == {"1step": 14_777_616, "2step": 5_760_000}
    assert p.resize_to_pixels == {"1step": 3200 * 1800, "2step": 0}
    assert p.crop == {"kind": "area", "portion": 616 ** 2, "ref_pixels": 5_760_000} and p.zoom == 3.0


def test_medspot_protocol_matches_strict_eval():
    p = load_protocol("medspot@v1")
    assert p.max_pixels == {"1step": 5_860_400, "2step": 5_860_400}
    assert p.crop == {"kind": "fixed", "px": 616} and p.zoom == 2.0
    assert "untuned-zoom" in p.flags


def test_models_cover_release_list():
    ids = {m.id for m in all_models()}
    assert {"ours-2b-ab", "ours-2b-ab-clin", "ours-4b-ab", "ours-4b-ab-clin", "ours-9b-ab", "ours-9b-ab-clin",
            "gui-aima-3b", "gui-actor-2b", "gui-actor-3b", "gui-actor-7b",
            "qwen35-2b-base", "qwen35-4b-base", "qwen35-9b-base"} <= ids


def test_bench_order_matches_configs():
    from benchev.config import BENCH_ORDER, all_benches
    assert sorted(BENCH_ORDER) == all_benches() and len(set(BENCH_ORDER)) == len(BENCH_ORDER)


def test_path_map_rewrites_ckpt_and_base(monkeypatch):
    monkeypatch.setenv("BENCHEV_PATH_MAP", "/nowhere/=/x/;checkpoints/=/mnt/ckpts/")
    assert load_model("ours-2b-ab-clin").ckpt == "/mnt/ckpts/ours-2b-ab-clin"
    monkeypatch.delenv("BENCHEV_PATH_MAP")
    assert load_model("ours-2b-ab-clin").ckpt == "checkpoints/ours-2b-ab-clin"


def test_gui_actor_ckpts_pinned():
    from benchev.config import load_model
    revs = {"gui-actor-2b": ("GUI-Actor-2B-Qwen2-VL", "8f87b366d004425a9823502553e2097c71116ece"),
            "gui-actor-3b": ("GUI-Actor-3B-Qwen2.5-VL", "5fb97348752cb3d50be9709f47d9ae6c99725949"),
            "gui-actor-7b": ("GUI-Actor-7B-Qwen2.5-VL", "b6296a5555f4becb5fba10ab60bd99de07466487")}
    for mid, (repo, rev) in revs.items():
        m = load_model(mid)
        assert m.adapter == "gui_actor" and m.ckpt.endswith(f"/{repo}") and rev in m.notes


def test_model_cfg_new_fields_default():
    from benchev.config import ModelCfg
    m = ModelCfg(id="x", kind="published", adapter="hf_chat", ckpt=None, base=None, status="published")
    assert (m.recipe, m.max_pixels, m.env, m.flags, m.parity) == (None, None, None, [], None)


MEDSPOT_COMPARISON = {   # id: (adapter, recipe, no-published-reference)
    "ui-venus-1.5-2b": ("hf_chat", "ui_venus", False), "mai-ui-2b": ("hf_chat", "mai_ui", False),
    "ui-tars-1.5-7b": ("hf_chat", "ui_tars15", False), "qwen2.5-vl-7b": ("hf_chat", "qwen25vl", False),
    "qwen3-vl-8b": ("hf_chat", "qwen3vl", False), "qwen2-vl-7b": ("hf_chat", "qwen2vl_k1000", True),
    "os-atlas-7b": ("hf_chat", "os_atlas", False), "uground-7b": ("hf_chat", "uground", False),
    "aguvis-7b": ("hf_chat", "aguvis", False), "gui-owl-7b": ("hf_chat", "gui_owl", False),
    "gui-owl-1.5-8b": ("hf_chat", "gui_owl15", False), "kv-ground-8b": ("hf_chat", "kv_ground", False),
    "ministral-3-8b": ("hf_chat", "legacy_coords", True), "llama-3.2-11b-vision": ("hf_chat", "legacy_coords", True),
    "deepseek-vl2-small": ("deepseek_vl2", "deepseek_ref", True), "cogagent-9b": ("cogagent", "cogagent", False),
    "seeclick": ("seeclick", "seeclick", False),
    "qwen3-vl-8b-sspro": ("hf_chat", "qwen3vl_sspro", False), "qwen2.5-vl-7b-sspro": ("hf_chat", "qwen25vl_sspro", False),
    "qwen3-vl-8b-legacy": ("hf_chat", "qwen3vl_legacy", True),
}


def test_new_model_configs_load():
    import re
    from benchev.adapters import get_adapter
    from benchev.adapters.recipes import RECIPES
    for mid, (adapter, recipe, noref) in MEDSPOT_COMPARISON.items():
        m = load_model(mid)
        assert (m.kind, m.adapter, m.recipe) == ("published", adapter, recipe), mid
        assert get_adapter(m.adapter).name == adapter and m.recipe in RECIPES
        assert m.env and m.params and re.search(r"\S+/\S+ @[0-9a-f]{40}", m.notes), mid
        assert ("no-published-reference" in m.flags) == noref, mid
