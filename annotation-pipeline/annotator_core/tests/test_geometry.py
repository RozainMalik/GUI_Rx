import pytest
from annotator import geometry as g

FULL = (0, 0, 1920, 1080)


def test_full_frame_round_trip():
    assert g.box_1000_to_px([0, 0, 1000, 1000], FULL) == (0, 0, 1920, 1080)
    assert g.px_to_1000((0, 0, 1920, 1080), FULL) == [0, 0, 1000, 1000]


@pytest.mark.parametrize("window", [(0, 0, 500, 300), (1420, 0, 1920, 300), (0, 780, 500, 1080), (1420, 780, 1920, 1080)])
def test_crop_mapping_at_all_four_corners(window):
    # the whole crop maps back onto its own window, whichever corner it sits in
    assert g.box_1000_to_px([0, 0, 1000, 1000], window) == window
    # a box in the crop's lower-right quadrant lands in the window's lower-right quadrant
    box = g.box_1000_to_px([500, 500, 1000, 1000], window)
    assert box == (window[0] + 250, window[1] + 150, window[2], window[3])
    assert g.px_to_1000(box, window) == [500, 500, 1000, 1000]


def test_px_to_1000_rounds_to_int():
    assert g.px_to_1000((288, 162, 576, 216), FULL) == [150, 150, 300, 200]


def test_one_row_off_does_not_agree():
    row = (100, 100, 300, 120)
    next_row = (100, 121, 300, 141)
    assert not g.agree(row, next_row)


def test_extent_difference_on_same_row_agrees():
    tight = (100, 100, 300, 120)
    with_gutter = (80, 98, 300, 122)
    assert g.agree(tight, with_gutter)


def test_agree_is_symmetric_and_needs_both_centres():
    big = (0, 0, 1000, 1000)
    small_in_corner = (10, 10, 30, 30)     # its centre is inside big, but big's centre is not inside it
    assert not g.agree(big, small_in_corner)
    assert not g.agree(small_in_corner, big)


def test_median_box_odd_and_even():
    assert g.median_box([(0, 0, 10, 10), (2, 2, 12, 12), (4, 4, 14, 14)]) == (2, 2, 12, 12)
    # even count takes the lower middle so the result is always an int box
    assert g.median_box([(0, 0, 10, 10), (2, 2, 12, 12)]) == (0, 0, 10, 10)
    with pytest.raises(ValueError):
        g.median_box([])


def test_clusters_largest_first():
    boxes = {"a": (100, 100, 300, 120), "b": (98, 100, 302, 121), "c": (100, 101, 300, 119), "d": (100, 300, 300, 320)}
    comps = g.clusters(boxes)
    assert comps[0] == ["a", "b", "c"]
    assert comps[1] == ["d"]


def test_clusters_are_connected_components_not_cliques():
    # a agrees with b, b agrees with c, a does not agree with c -> still one component
    a, b, c = (100, 100, 220, 120), (150, 100, 270, 120), (200, 100, 320, 120)
    assert g.agree(a, b) and g.agree(b, c) and not g.agree(a, c)
    assert g.clusters({"a": a, "b": b, "c": c}) == [["a", "b", "c"]]


def test_percent_xywh_matches_benchmark_inverse():
    box = (291, 302, 576, 328)
    pct = g.px_to_pct_xywh(box, 1920, 1080)
    assert pct == [15.16, 27.96, 14.84, 2.41]
    back = g.pct_xywh_to_px_xyxy(pct, 1920, 1080)
    for got, want in zip(back, box):
        assert abs(got - want) < 0.2   # two decimals of percent is < 0.2 px on 1920


def test_predicates():
    assert g.is_degenerate((5, 5, 5, 9)) and not g.is_degenerate((0, 0, 1, 1))
    assert g.area((0, 0, 10, 5)) == 50 and g.size((2, 3, 12, 8)) == (10, 5)
    assert g.center((0, 0, 10, 10)) == (5.0, 5.0)
    assert g.contains_point((0, 0, 10, 10), (9.9, 0)) and not g.contains_point((0, 0, 10, 10), (10, 0))
    assert g.inside((1, 1, 9, 9), (0, 0, 10, 10)) and not g.inside((-1, 1, 9, 9), (0, 0, 10, 10))
    assert g.inside((-1, 1, 9, 9), (0, 0, 10, 10), slack=1)
    assert g.intersects((0, 0, 10, 10), (9, 9, 20, 20)) and not g.intersects((0, 0, 10, 10), (10, 10, 20, 20))
    assert g.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0 and g.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert g.clamp_box((-5, -5, 2000, 1100), 1920, 1080) == (0, 0, 1920, 1080)


from annotator.geometry import COORD_MODES, CoordSpace, full_space

CORNERS = [(0, 0, 500, 300), (1420, 0, 1920, 300), (0, 780, 500, 1080), (1420, 780, 1920, 1080)]


@pytest.mark.parametrize("window", CORNERS)
def test_coord_space_pixels_mode_maps_the_sent_image_onto_its_window(window):
    shown = (1000, 600)                                  # the crop was upscaled 2x before it was sent
    space = CoordSpace(window, shown, "pixels")
    assert space.bounds() == (1000, 600)
    assert space.to_px([0, 0, 1000, 600]) == window
    box = space.to_px([500, 300, 1000, 600])
    assert box == (window[0] + 250, window[1] + 150, window[2], window[3])
    assert space.from_px(box) == [500, 300, 1000, 600]


@pytest.mark.parametrize("window", CORNERS)
def test_coord_space_thousand_mode_is_box_1000_to_px(window):
    space = CoordSpace(window, (123, 456), "thousand")   # the sent size is irrelevant on the 0-1000 scale
    assert space.bounds() == (1000, 1000)
    for b in ([0, 0, 1000, 1000], [500, 500, 1000, 1000], [10, 20, 30, 40]):
        assert space.to_px(b) == g.box_1000_to_px(b, window)
        assert space.from_px(space.to_px(b)) == g.px_to_1000(space.to_px(b), window)


def test_full_space_default_mode_and_bad_mode():
    s = full_space(1920, 1080, "pixels")
    assert s.window == (0, 0, 1920, 1080) and s.shown_size == (1920, 1080)
    assert s.to_px([288, 162, 576, 216]) == (288, 162, 576, 216) and s.from_px((288, 162, 576, 216)) == [288, 162, 576, 216]
    assert full_space(1920, 1080).mode == "thousand" and COORD_MODES == ("thousand", "pixels")
    with pytest.raises(ValueError):
        CoordSpace(FULL, (1920, 1080), "percent")
