from benchev.geometry import point_in_box, point_in_polygon, resize_to_area, area_crop_px, crop_window


def test_point_in_box_edges_inclusive():
    assert point_in_box((0.1, 0.2), (0.1, 0.2, 0.3, 0.4))
    assert not point_in_box((0.31, 0.3), (0.1, 0.2, 0.3, 0.4))
    assert not point_in_box(None, (0, 0, 1, 1))


def test_point_in_polygon_square_and_concave():
    sq = ((0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3))
    assert point_in_polygon((0.2, 0.2), sq) and not point_in_polygon((0.4, 0.2), sq)
    L = ((0, 0), (0.4, 0), (0.4, 0.1), (0.1, 0.1), (0.1, 0.4), (0, 0.4))
    assert point_in_polygon((0.05, 0.3), L) and not point_in_polygon((0.3, 0.3), L)
    assert not point_in_polygon(None, sq)


def test_resize_to_area_matches_port():
    assert resize_to_area(1920, 1080, 0) == (1920, 1080)
    assert resize_to_area(1920, 1080, 3200 * 1800) == (3200, 1800)
    assert resize_to_area(3840, 2160, 3200 * 1800) == (3200, 1800)


def test_area_crop_px_matches_port_sspro_crop_px():
    # gui_aima.qwen35.inference.sspro_crop_px(3840, 2160) with multiple=32
    assert area_crop_px(3840, 2160, 616 ** 2, 5_760_000, 32) == int((3840 * 2160 * 616 ** 2 / 5_760_000) ** 0.5 / 32) * 32
    assert area_crop_px(10, 10, 616 ** 2, 5_760_000, 32) == 32


def test_crop_window_matches_ground_two_step():
    assert crop_window(1920, 1080, (0.5, 0.5), 616) == (652, 232, 1268, 848)
    assert crop_window(1920, 1080, (0.0, 0.0), 616) == (0, 0, 616, 616)
    assert crop_window(1920, 1080, (1.0, 1.0), 616) == (1304, 464, 1920, 1080)
    assert crop_window(500, 400, (0.5, 0.5), 616) == (0, 0, 500, 400)
