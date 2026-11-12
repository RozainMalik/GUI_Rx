import annotator
from tests.conftest import make_frame, FakeGenerator, run


def test_version():
    assert annotator.PIPELINE_VERSION == "1.0.0"


def test_frame_fixture_is_full_hd(frame):
    assert frame.size == (1920, 1080)
    assert frame.getpixel((250, 130)) == (200, 200, 210)


def test_fake_generator_pops_in_order():
    from annotator.calling import Call
    gen = FakeGenerator(["one", "two"])
    call = Call(system="s", user_text="u")
    assert run(gen.generate(call)).content == "one"
    assert run(gen.generate(call)).content == "two"
    assert len(gen.calls) == 2
