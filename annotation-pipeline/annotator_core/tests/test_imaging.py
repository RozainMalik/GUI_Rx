from annotator import imaging as im
from annotator.imaging import make_crop
from tests.conftest import make_frame

W, H = 1920, 1080


def test_region_window_pads_by_five_percent_and_floors():
    # region 768 x 324 -> pad 38.4 x 16.2 -> floor on the near edges, ceil on the far edges
    assert im.region_window((192, 108, 960, 432), W, H) == (153, 91, 999, 449)


def test_region_window_clamps_to_frame():
    assert im.region_window((0, 0, 1000, 1000), W, H) == (0, 0, 1050, 1050)
    assert im.region_window((1800, 1000, 1920, 1080), W, H)[2:] == (1920, 1080)


def test_verify_window_uses_max_of_box_and_floor_padding():
    # small box: floors apply (200 px horizontally, 120 px vertically)
    assert im.verify_window((500, 500, 560, 520), W, H) == (300, 380, 760, 640)
    # big box: its own size applies (w horizontally, 2h vertically)
    assert im.verify_window((500, 500, 900, 700), W, H) == (100, 100, 1300, 1080)   # y2 clamped to the frame


def test_focus_windows_have_frame_aspect_and_slide_not_shrink():
    wins = im.focus_windows((960, 540), W, H)
    assert [tuple(map(lambda v: v, w)) for w in wins] == [(480, 270, 1440, 810), (624, 351, 1296, 729), (720, 405, 1200, 675)]
    corner = im.focus_windows((5, 5), W, H)
    assert corner[0] == (0, 0, 960, 540)                     # slid to the origin, still 960x540
    far = im.focus_windows((1915, 1075), W, H)
    assert far[2] == (1440, 810, 1920, 1080)                # slid to the far corner, still 480x270


def test_upscale_factor_targets_1024_and_clamps():
    assert im.upscale_factor((0, 0, 512, 100)) == 2.0
    assert im.upscale_factor((0, 0, 100, 50)) == 4.0
    assert im.upscale_factor((0, 0, 2000, 100)) == 1.0
    assert im.upscale_factor((0, 0, 1024, 700)) == 1.0


def test_make_crop_records_window_scale_and_shown_size():
    crop = im.make_crop(make_frame(), (100, 100, 612, 356))    # 512 x 256 -> x2
    assert crop.window == (100, 100, 612, 356)
    assert crop.scale == 2.0
    assert crop.shown_size == (1024, 512)
    assert crop.image.size == (1024, 512)


def test_box_to_crop_maps_frame_px_into_the_upscaled_crop():
    crop = im.make_crop(make_frame(), (100, 100, 612, 356))
    assert im.box_to_crop((110, 120, 210, 170), crop) == (20, 40, 220, 140)


def test_draw_box_is_red_three_px_on_a_copy():
    frame = make_frame()
    out = im.draw_box(frame, (300, 200, 400, 240))
    assert out is not frame
    assert out.getpixel((300, 200)) == (255, 0, 0)
    assert out.getpixel((302, 220)) == (255, 0, 0)          # third pixel of the left edge
    assert out.getpixel((303, 220)) != (255, 0, 0)          # fourth is the original
    assert out.getpixel((399, 239)) == (255, 0, 0)          # x2/y2 are exclusive: last drawn pixel is 399,239
    assert frame.getpixel((300, 200)) != (255, 0, 0)        # original untouched


def test_touches_edge_only_on_interior_crop_edges():
    window = (100, 100, 600, 400)
    assert im.touches_edge((101, 200, 300, 250), window, W, H)          # 1 px from the left edge
    assert im.touches_edge((200, 200, 599, 250), window, W, H)          # 1 px from the right edge
    assert not im.touches_edge((150, 200, 300, 250), window, W, H)
    at_frame_border = (0, 100, 600, 400)
    assert not im.touches_edge((1, 200, 300, 250), at_frame_border, W, H)   # the frame's own edge is not truncation


def test_crop_space_uses_the_window_and_the_upscaled_size():
    crop = make_crop(make_frame(), (192, 108, 960, 432))
    s = crop.space("pixels")
    assert s.window == crop.window and s.shown_size == crop.image.size and s.mode == "pixels"
    assert s.to_px([0, 0, *crop.image.size]) == crop.window
    assert crop.space().mode == "thousand"
