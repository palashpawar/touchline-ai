"""Overlays: projected pitch template, player boxes/trails, top-down minimap, annotated video."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import calibration as cal

TEAM_BGR = {"A": (245, 245, 245), "B": (200, 120, 30), "other": (0, 200, 255), "unknown": (160, 160, 160)}


def pitch_polylines(L: float = 105.0, W: float = 68.0) -> list[np.ndarray]:
    def rect(x0, y0, x1, y1):
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], np.float64)

    def arc(cx, cy, r, a0, a1, n=48):
        a = np.radians(np.linspace(a0, a1, n))
        return np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]

    c = W / 2
    lines = [rect(0, 0, L, W), np.array([[L / 2, 0], [L / 2, W]]), arc(L / 2, c, 9.15, 0, 360, 96)]
    for gx, s in ((0.0, 1), (L, -1)):
        lines.append(rect(gx, c - 20.16, gx + s * 16.5, c + 20.16))
        lines.append(rect(gx, c - 9.16, gx + s * 5.5, c + 9.16))
        lines.append(arc(gx + s * 11, c, 9.15, -53 if s > 0 else 127, 53 if s > 0 else 233))
    return [_densify(p) for p in lines]


def _densify(p: np.ndarray, step: float = 1.0) -> np.ndarray:
    out = [p[0]]
    for a, b in zip(p, p[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step))
        out += [a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:]]
    return np.asarray(out)


def draw_pitch_projection(img: np.ndarray, H_img_to_pitch: np.ndarray, pitch_size=(105.0, 68.0),
                          color=(255, 0, 255), thickness: int = 2, scale: float = 1.0) -> None:
    """Draw the pitch template through the inverse homography (only points in front of the camera)."""
    Hinv = np.linalg.inv(np.asarray(H_img_to_pitch, np.float64))
    for poly in pitch_polylines(*pitch_size):
        ph = np.c_[poly, np.ones(len(poly))] @ Hinv.T
        ok = ph[:, 2] > 1e-9 if np.median(ph[:, 2]) > 0 else ph[:, 2] < -1e-9
        uv = ph[:, :2] / ph[:, 2:3]
        h, w = img.shape[:2]
        ok &= (uv[:, 0] > -w) & (uv[:, 0] < 2 * w) & (uv[:, 1] > -h) & (uv[:, 1] < 2 * h)
        run = []
        for good, p in zip(ok, uv * scale):
            if good:
                run.append(p)
            elif len(run) > 1:
                cv2.polylines(img, [np.int32(run)], False, color, thickness, cv2.LINE_AA)
                run = []
            else:
                run = []
        if len(run) > 1:
            cv2.polylines(img, [np.int32(run)], False, color, thickness, cv2.LINE_AA)


def minimap(players: list[dict], pitch_size=(105.0, 68.0), px_per_m: float = 4.0, trails=None,
            visible_poly: np.ndarray | None = None) -> np.ndarray:
    L, W = pitch_size
    pad = 8
    w, h = int(L * px_per_m) + 2 * pad, int(W * px_per_m) + 2 * pad
    img = np.full((h, w, 3), (40, 92, 38), np.uint8)
    for k in range(0, int(L), 10):
        if (k // 10) % 2 == 0:
            cv2.rectangle(img, (pad + int(k * px_per_m), pad), (pad + int((k + 10) * px_per_m), h - pad), (44, 100, 42), -1)
    if visible_poly is not None and len(visible_poly) >= 3:
        over = img.copy()
        cv2.fillPoly(over, [np.int32(np.asarray(visible_poly) * px_per_m + pad)], (60, 130, 60))
        img = cv2.addWeighted(over, 0.6, img, 0.4, 0)
    for poly in pitch_polylines(L, W):
        cv2.polylines(img, [np.int32(poly * px_per_m + pad)], False, (230, 230, 230), 1, cv2.LINE_AA)
    for tr in trails or []:
        pts = np.int32(np.asarray(tr["points"]) * px_per_m + pad)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, TEAM_BGR.get(tr["team"], (160, 160, 160)), 1, cv2.LINE_AA)
    for p in players:
        c = (int(pad + p["x"] * px_per_m), int(pad + p["y"] * px_per_m))
        cv2.circle(img, c, 5, TEAM_BGR.get(p["team"], (160, 160, 160)), -1, cv2.LINE_AA)
        cv2.circle(img, c, 5, (20, 20, 20), 1, cv2.LINE_AA)
    return img


def open_writer(path: Path, fps: float, size: tuple[int, int]):
    """VP8/WebM plays in all major browsers and is available in the OpenCV 5 wheels' FFmpeg."""
    path = Path(path)
    for fourcc, ext in (("VP80", ".webm"), ("avc1", ".mp4"), ("mp4v", ".mp4")):
        out = path.with_suffix(ext)
        w = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if w.isOpened():
            return w, out
    raise RuntimeError("no usable video encoder in this OpenCV build")
