import sys

from benchev.adapters import get_adapter
from benchev.adapters.gui_aima_upstream import UPSTREAM, conversation, use_upstream_package


def test_registered():
    a = get_adapter("gui_aima_upstream")
    assert a.name == "gui_aima_upstream" and a.crop_multiple == 28


def test_upstream_package_shadows_qwen35_port():
    import gui_aima  # noqa: F401  (our port, via vendor/)
    use_upstream_package()
    try:
        import gui_aima
        assert gui_aima.__file__.startswith(UPSTREAM)
    finally:
        sys.path.remove(UPSTREAM)
        for m in [m for m in sys.modules if m == "gui_aima" or m.startswith("gui_aima.")]:
            del sys.modules[m]


def test_conversation_matches_upstream_layout():
    c = conversation("IMG", "click ok")
    assert [m["role"] for m in c] == ["system", "user"]
    assert c[1]["content"] == [{"type": "image", "image": "IMG"}, {"type": "text", "text": "click ok"}]
    assert c[0]["content"][0]["text"].startswith("You are a GUI agent.")


def test_last_position_head_matches_full_head_on_last_position():
    torch = __import__("pytest").importorskip("torch")
    from benchev.adapters.gui_aima_upstream import last_position_head
    full = torch.nn.Linear(8, 11, bias=False)
    h = torch.randn(1, 5, 8)
    head = last_position_head(full)
    out = head(h)
    assert out.shape == (1, 1, 11)
    assert torch.allclose(out[:, -1, :], full(h)[:, -1, :], atol=1e-6)
    assert head.weight is full.weight
