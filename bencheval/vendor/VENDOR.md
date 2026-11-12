# Vendored code

| Path | Source | Commit |
|---|---|---|
| `gui_aima/` | the Qwen3.5 port of GUI-AIMA in `../training/src/gui_aima` (plus `qwen35/text_coord.py` and `qwen35/probe.py`, used by the zero-shot base-model adapter) | this release |
| `medspot_strict/core.py` | the MedSPOT strict scorer, unchanged (identical to `../training/src/medspot_strict/core.py`) | this release |
| `medspot_metrics.py` | `../training/src/attn_ft/medspot_metrics.py` | this release |
| `upstream_gui_aima/gui_aima/` + LICENSE (Apache-2.0) | github.com/sjz5202/GUI-AIMA `src/gui_aima` (official Qwen2.5-VL code) | a0b938e8d26dd746d2b6b3aa29d63fd6b463c54c |
| `upstream_gui_actor/gui_actor/` + LICENSE (MIT) | github.com/microsoft/GUI-Actor `src/gui_actor` (official Qwen2-VL / Qwen2.5-VL code) | d98d1bbd01862f9112114b83b032f492c365a173 |

The upstream folders are byte-identical to their sources (`upstream_gui_actor/SHA256SUMS` is checked by the tests).
