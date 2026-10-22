"""Team assignment from jersey colour: a heuristic with exposed uncertainty.

Feature: median CIE-Lab colour of the non-grass torso pixels of each observation.
Per track we take the median over its observations. Clustering: ``cv2.kmeans`` with
K=3 over track features (two teams plus "other": goalkeepers, referees, mis-detections);
the two most populated clusters are the teams. If the user supplies team colours, we
assign to the nearest supplied colour instead and mark tracks farther than
``max_lab_dist`` as "other".

Confidence (0..1) combines (a) the per-observation vote share for the chosen team and
(b) the relative margin between nearest and second-nearest team centre. It is a
heuristic agreement score, NOT a calibrated probability.
"""
from __future__ import annotations

import cv2
import numpy as np


def torso_lab(frame: np.ndarray, box) -> list[float] | None:
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = y2 - y1, x2 - x1
    if h < 8 or w < 4:
        return None
    crop = frame[max(0, y1 + int(0.15 * h)): max(0, y1 + int(0.5 * h)), max(0, x1 + int(0.25 * w)): max(0, x2 - int(0.25 * w))]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    keep = cv2.inRange(hsv, (30, 50, 35), (90, 255, 255)) == 0
    if keep.sum() < 4:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab).reshape(-1, 3)[keep.reshape(-1)]
    return np.median(lab.astype(np.float32), axis=0).tolist()


def hex_to_lab(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    rgb = np.array([[[int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)]]], np.uint8)  # BGR
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2Lab).reshape(3).astype(np.float32)


def lab_to_hex(lab) -> str:
    px = np.clip(np.array([[lab]], np.float32), 0, 255).astype(np.uint8)
    b, g, r = cv2.cvtColor(px, cv2.COLOR_Lab2BGR).reshape(3)
    return f"#{r:02x}{g:02x}{b:02x}"


def assign_teams(track_obs_feats: dict[int, list[list[float]]], team_colors: list[str] | None = None,
                 max_lab_dist: float = 60.0, overrides: dict[int, str] | None = None) -> dict:
    """Return {"tracks": {tid: {team, confidence, feature}}, "centres": {...}, "method": str}."""
    feats = {tid: np.median(np.asarray(f, np.float32), axis=0) for tid, f in track_obs_feats.items() if f}
    out = {"tracks": {}, "centres": {}, "method": ""}
    if not feats:
        out["method"] = "none"
        return out
    tids = list(feats)
    X = np.stack([feats[t] for t in tids]).astype(np.float32)
    if team_colors and len(team_colors) == 2:
        centres = np.stack([hex_to_lab(c) for c in team_colors])
        out["method"] = "user_colours"
    elif len(X) >= 3:
        k = 3 if len(X) >= 6 else 2
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        cv2.setRNGSeed(7)
        _, labels, C = cv2.kmeans(X, k, None, crit, 8, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.reshape(-1), minlength=k)
        order = np.argsort(-counts)[:2]
        centres = C[order]
        # Stable naming: team A is the lighter kit.
        if centres[0][0] < centres[1][0]:
            centres = centres[::-1]
        out["method"] = f"kmeans_k{k}"
    else:
        out["method"] = "insufficient_tracks"
        for t in tids:
            out["tracks"][t] = {"team": "unknown", "confidence": 0.0, "feature": feats[t].tolist()}
        return out
    out["centres"] = {"A": lab_to_hex(centres[0]), "B": lab_to_hex(centres[1])}
    names = ["A", "B"]
    for t in tids:
        d = np.linalg.norm(centres - feats[t], axis=1)
        j = int(np.argmin(d))
        margin = float((d[1 - j] - d[j]) / max(1e-6, d[1 - j] + d[j]))
        obs = np.asarray(track_obs_feats[t], np.float32)
        od = np.linalg.norm(obs[:, None, :] - centres[None], axis=2)
        vote = float(np.mean(np.argmin(od, axis=1) == j))
        team = names[j]
        # Far from both kits -> goalkeeper/referee/other. In k-means mode, use the
        # within-cluster spread as the yardstick.
        if d[j] > max_lab_dist:
            team = "other"
        conf = round(max(0.0, min(1.0, vote * (0.5 + margin))), 3)
        out["tracks"][t] = {"team": team, "confidence": conf if team != "other" else 0.0, "feature": feats[t].tolist()}
    for t, team in (overrides or {}).items():
        if t in out["tracks"]:
            out["tracks"][t].update(team=team, confidence=1.0, overridden=True)
    return out
