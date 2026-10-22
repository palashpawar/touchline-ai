import numpy as np

from touchline.vision.detector import Detection, filter_players


def scene():
    """Green pitch on top, blue track at the bottom; the (smoothed) field mask wrongly covers both."""
    img = np.zeros((400, 600, 3), np.uint8)
    img[:300] = (40, 140, 40)  # grass (BGR)
    img[300:] = (160, 80, 30)  # blue track
    fmask = np.full((400, 600), 255, np.uint8)  # leaky mask: everything "grass"
    return img, fmask


def test_raw_grass_under_foot_rejects_track_standers():
    img, fm = scene()
    player = Detection(np.array([100, 200, 120, 260.0]), 0.9)  # feet on grass
    photographer = Detection(np.array([300, 280, 322, 340.0]), 0.9)  # feet on the track
    kept, reasons = filter_players([player, photographer], fm, frame=img)
    assert kept == [player]
    assert reasons["no_grass_under_foot"] == 1


def test_without_frame_falls_back_to_mask_only():
    img, fm = scene()
    dets = [Detection(np.array([100, 200, 120, 260.0]), 0.9), Detection(np.array([300, 280, 322, 340.0]), 0.9)]
    kept, _ = filter_players(dets, fm)
    assert len(kept) == 2


def test_oversized_near_camera_person_rejected():
    img, fm = scene()
    dets = [Detection(np.array([50 + 30 * i, 200, 70 + 30 * i, 260.0]), 0.9) for i in range(5)]
    dets.append(Detection(np.array([400, 0, 560, 250.0]), 0.9))  # 250 px tall vs 60 px median, feet on grass
    kept, reasons = filter_players(dets, fm, frame=img)
    assert len(kept) == 5 and reasons["too_large"] == 1
