"""Multi-object tracking of players by detection association.

Design (deliberately simple and inspectable):
* State per track: ``cv2.KalmanFilter`` with constant-velocity model on the ground-contact
  point (x, y, vx, vy) in *reference-frame* image coordinates. When camera-motion
  compensation is on, detections are warped into the reference frame first, so the
  camera pan is not mistaken for player motion.
* Association: Hungarian assignment (scipy ``linear_sum_assignment``) on a cost that mixes
  position residual normalised by box height and torso-colour histogram distance
  (``cv2.compareHist``, Bhattacharyya). Pairs above the gate are left unmatched.
* Unmatched detections start *tentative* tracks, confirmed after 2 hits.
* Unmatched tracks coast on the Kalman prediction for up to ``max_coast_s`` seconds
  (temporary occlusion), then terminate. A player re-appearing after that gets a new ID:
  we do not do re-identification, so identity switches and re-spawned IDs are expected
  when players cross or leave the view.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Observation:
    frame: int
    t: float
    box: list[float]  # raw image box x1,y1,x2,y2
    foot_ref: list[float]  # ground point in reference-frame image coords
    score: float
    team_feat: list[float] | None = None
    pitch: list[float] | None = None  # metres, filled after projection


@dataclass
class Track:
    tid: int
    kf: cv2.KalmanFilter
    last_t: float
    height: float
    hist: np.ndarray | None
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    alive: bool = True
    obs: list[Observation] = field(default_factory=list)
    coasted_frames: list[int] = field(default_factory=list)


def torso_hist(frame: np.ndarray, box) -> np.ndarray | None:
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = y2 - y1, x2 - x1
    if h < 8 or w < 4:
        return None
    crop = frame[max(0, y1 + int(0.15 * h)): max(0, y1 + int(0.55 * h)), max(0, x1 + int(0.2 * w)): max(0, x2 - int(0.2 * w))]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (30, 50, 35), (90, 255, 255))
    hist = cv2.calcHist([hsv], [0, 1], cv2.bitwise_not(grass), [12, 6], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
    return hist


def _make_kf(x: float, y: float) -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.processNoiseCov = np.diag([4.0, 4.0, 60.0, 60.0]).astype(np.float32)
    kf.measurementNoiseCov = np.diag([9.0, 9.0]).astype(np.float32)
    kf.errorCovPost = np.diag([25.0, 25.0, 400.0, 400.0]).astype(np.float32)
    kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
    return kf


def _predict(kf: cv2.KalmanFilter, dt: float) -> np.ndarray:
    kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    return kf.predict()[:2].reshape(-1)


class Tracker:
    def __init__(self, gate: float = 1.6, color_weight: float = 0.35, max_coast_s: float = 1.0, min_hits: int = 2):
        self.gate, self.cw, self.max_coast_s, self.min_hits = gate, color_weight, max_coast_s, min_hits
        self.tracks: list[Track] = []
        self.next_id = 1
        self.stats = {"matched": 0, "new_tracks": 0, "coasted": 0, "terminated": 0, "assoc_residual_h": []}

    def step(self, frame_idx: int, t: float, detections: list[dict]) -> None:
        """detections: dicts with keys box, foot_ref, score, height, hist, team_feat."""
        live = [tr for tr in self.tracks if tr.alive]
        preds = [_predict(tr.kf, max(1e-3, t - tr.last_t)) for tr in live]
        n, m = len(live), len(detections)
        cost = np.full((n, m), 1e6)
        resid = np.full((n, m), np.inf)
        for i, tr in enumerate(live):
            for j, d in enumerate(detections):
                hh = max(8.0, 0.5 * (tr.height + d["height"]))
                r = float(np.linalg.norm(preds[i] - np.asarray(d["foot_ref"]))) / hh
                resid[i, j] = r
                if r > self.gate:
                    continue
                c = r / self.gate
                if tr.hist is not None and d["hist"] is not None:
                    c += self.cw * cv2.compareHist(tr.hist, d["hist"], cv2.HISTCMP_BHATTACHARYYA)
                cost[i, j] = c
        matched_t, matched_d = set(), set()
        if n and m:
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] >= 1e6:
                    continue
                tr, d = live[i], detections[j]
                tr.kf.correct(np.asarray(d["foot_ref"], np.float32).reshape(2, 1))
                tr.height = 0.7 * tr.height + 0.3 * d["height"]
                if d["hist"] is not None:
                    tr.hist = d["hist"] if tr.hist is None else (0.8 * tr.hist + 0.2 * d["hist"]).astype(np.float32)
                tr.last_t, tr.hits, tr.misses = t, tr.hits + 1, 0
                tr.confirmed = tr.confirmed or tr.hits >= self.min_hits
                tr.obs.append(self._obs(frame_idx, t, d))
                matched_t.add(i)
                matched_d.add(j)
                self.stats["matched"] += 1
                self.stats["assoc_residual_h"].append(float(resid[i, j]))
        for i, tr in enumerate(live):
            if i in matched_t:
                continue
            tr.misses += 1
            if t - tr.last_t > self.max_coast_s or (not tr.confirmed and tr.misses >= 2):
                tr.alive = False
                self.stats["terminated"] += 1
            else:
                tr.coasted_frames.append(frame_idx)
                self.stats["coasted"] += 1
        for j, d in enumerate(detections):
            if j in matched_d:
                continue
            tr = Track(self.next_id, _make_kf(*d["foot_ref"]), t, d["height"], d["hist"])
            tr.obs.append(self._obs(frame_idx, t, d))
            self.tracks.append(tr)
            self.next_id += 1
            self.stats["new_tracks"] += 1

    @staticmethod
    def _obs(frame_idx, t, d) -> Observation:
        pitch = d.get("pitch")
        return Observation(frame_idx, t, [float(v) for v in d["box"]], [float(v) for v in d["foot_ref"]],
                           float(d["score"]), d.get("team_feat"),
                           None if pitch is None else [float(pitch[0]), float(pitch[1])])

    def finished(self) -> list[Track]:
        return [tr for tr in self.tracks if tr.confirmed or tr.hits >= self.min_hits]
