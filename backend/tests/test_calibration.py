import numpy as np
import pytest

from touchline.vision import calibration as cal

SHAPE = (1080, 1920)
# A plausible sideline view of the left penalty area: far corners higher and closer together.
IMG = [[700, 520], [1150, 530], [1500, 760], [380, 740]]
PIT = cal.rectangle_pitch_points(0, 13.84, 16.5, 40.32)


def test_rectangle_points_are_in_cyclic_order():
    p = cal.rectangle_pitch_points(1, 2, 10, 5)
    assert p.tolist() == [[1, 2], [11, 2], [11, 7], [1, 7]]


def test_valid_calibration_maps_points_exactly():
    rep = cal.validate_calibration(IMG, PIT, SHAPE)
    assert rep.valid, rep.errors
    H = np.asarray(rep.homography)
    np.testing.assert_allclose(cal.to_pitch(H, IMG), PIT, atol=1e-6)
    # Round trip pitch -> image -> pitch
    q = cal.to_image(H, [[8.0, 30.0]])
    np.testing.assert_allclose(cal.to_pitch(H, q), [[8.0, 30.0]], atol=1e-6)


def test_interior_point_projects_inside_rectangle():
    rep = cal.validate_calibration(IMG, PIT, SHAPE)
    centre_img = np.mean(IMG, axis=0)
    x, y = cal.to_pitch(np.asarray(rep.homography), [centre_img])[0]
    assert 0 < x < 16.5 and 13.84 < y < 54.16


def test_swapped_points_rejected_as_out_of_order():
    bad = [IMG[0], IMG[2], IMG[1], IMG[3]]  # bow-tie
    rep = cal.validate_calibration(bad, PIT, SHAPE)
    assert not rep.valid
    assert any("order" in e for e in rep.errors)
    # the ordering problem is the only (and first) message, not a misleading "area too small"
    assert not any("area" in e for e in rep.errors if e.startswith("image points"))


def test_mirrored_order_rejected():
    mirrored = [IMG[1], IMG[0], IMG[3], IMG[2]]  # same quad traversed the other way relative to the pitch
    rep = cal.validate_calibration(mirrored, PIT, SHAPE)
    assert not rep.valid
    assert any("mirrored" in e for e in rep.errors)


@pytest.mark.parametrize("pts,needle", [
    ([[100, 100], [100, 100], [500, 500], [100, 500]], "apart"),
    ([[100, 100], [500, 100], [900, 100], [500, 103]], "collinear"),
    ([[100, 100], [105, 100], [105, 104], [100, 104]], "too small"),
])
def test_degenerate_image_points_rejected(pts, needle):
    rep = cal.validate_calibration(pts, PIT, SHAPE)
    assert not rep.valid
    assert any(needle in e for e in rep.errors), rep.errors


def test_wrong_point_count_rejected():
    rep = cal.validate_calibration(IMG[:3], PIT[:3], SHAPE)
    assert not rep.valid


def test_points_outside_frame_rejected():
    pts = [[700, 520], [1150, 530], [2500, 760], [380, 740]]
    rep = cal.validate_calibration(pts, PIT, SHAPE)
    assert not rep.valid and any("outside the frame" in e for e in rep.errors)


def test_holdout_error_reported():
    H = np.asarray(cal.validate_calibration(IMG, PIT, SHAPE).homography)
    img_spot = cal.to_image(H, [[11.0, 34.0]])[0]
    rep = cal.validate_calibration(IMG, PIT, SHAPE, holdout=[
        {"name": "spot", "image": img_spot.tolist(), "pitch": [11.0, 34.0]},
        {"name": "off", "image": (img_spot + [40, 0]).tolist(), "pitch": [11.0, 34.0]}])
    errs = {h["name"]: h["error_m"] for h in rep.metrics["holdout"]}
    assert errs["spot"] < 1e-6 and errs["off"] > 0.5


def test_visibility_reports_partial_view():
    rep = cal.validate_calibration(IMG, PIT, SHAPE)
    m = rep.metrics
    assert 0 < m["visible_pitch_fraction"] < 1
    assert m["visible_width_m"] <= 68.0 + 1e-6


def test_distance_outside_polygon():
    quad = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    d = cal.distance_outside_polygon(np.array([[5, 5], [15, 5], [5, -3]]), quad)
    np.testing.assert_allclose(d, [0, 5, 3], atol=1e-6)


def test_standard_landmarks_are_consistent():
    lm = cal.standard_landmarks()
    assert lm["left_penalty_area_near_front"] == (16.5, 54.16)
    assert lm["right_penalty_area_near_front"] == (105 - 16.5, 54.16)
    assert lm["centre_spot"] == (52.5, 34.0)
