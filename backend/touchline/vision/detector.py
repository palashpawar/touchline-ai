"""Person detection with a pretrained YOLOX-S ONNX model executed by OpenCV's DNN module.

Model: OpenCV Zoo ``object_detection_yolox_2022nov.onnx`` (YOLOX-S, COCO, Apache-2.0),
pinned by Hugging Face revision and SHA-256 in models/MODEL_CARD.md. The detector is
an external pretrained network; OpenCV contributes the inference runtime (cv2.dnn),
pre/post-processing and NMS, not the learned weights.

Broadcast-style wide shots make players ~30-60 px tall, too small for a single 640x640
pass over a 1920 px frame, so we run overlapping 640 px tiles over the grass bounding box.
"""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
PERSON = 0


@dataclass
class Detection:
    box: np.ndarray  # x1,y1,x2,y2 (px)
    score: float

    @property
    def foot(self) -> np.ndarray:
        """Bottom-centre of the box: approximate ground-contact point."""
        return np.array([(self.box[0] + self.box[2]) / 2, self.box[3]])

    @property
    def height(self) -> float:
        return float(self.box[3] - self.box[1])


class YoloxPersonDetector:
    _lock = threading.Lock()

    def __init__(self, model_path: Path, conf: float = 0.30, nms: float = 0.45, input_size: int = 640,
                 verify_hash: bool = True):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"detector weights missing at {model_path}; run scripts/fetch_model.sh")
        if verify_hash:
            digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
            if digest != MODEL_SHA256:
                raise ValueError(f"detector weights hash mismatch: {digest}")
        self.net = cv2.dnn.readNet(str(model_path))
        self.conf, self.nms, self.S = conf, nms, input_size
        grids, strides = [], []
        for s in (8, 16, 32):
            n = input_size // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            g = np.stack((xv, yv), 2).reshape(-1, 2)
            grids.append(g)
            strides.append(np.full((len(g), 1), s))
        self.grids = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(strides).astype(np.float32)

    def _infer(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        r = min(self.S / h, self.S / w)
        pad = np.full((self.S, self.S, 3), 114, np.uint8)
        rs = cv2.resize(img, (int(round(w * r)), int(round(h * r))), interpolation=cv2.INTER_LINEAR)
        pad[: rs.shape[0], : rs.shape[1]] = rs
        blob = np.transpose(pad, (2, 0, 1))[None].astype(np.float32)
        with self._lock:  # cv2.dnn.Net is not re-entrant
            self.net.setInput(blob)
            out = self.net.forward()
        d = np.array(out[0], np.float32)
        d[:, :2] = (d[:, :2] + self.grids) * self.strides
        d[:, 2:4] = np.exp(d[:, 2:4]) * self.strides
        score = d[:, 4] * d[:, 5 + PERSON]
        m = score > self.conf
        b = d[m, :4] / r
        xyxy = np.c_[b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2]
        return xyxy, score[m]

    def detect(self, frame: np.ndarray, roi: tuple[int, int, int, int] | None = None,
               tile: int = 640, overlap: int = 96) -> list[Detection]:
        H, W = frame.shape[:2]
        x0, y0, x1, y1 = roi if roi else (0, 0, W, H)
        # Extend the ROI upward so players standing on the far touchline are not cut at the head.
        y0 = max(0, y0 - 60)
        boxes, scores = [], []
        step = tile - overlap
        xs = list(range(x0, max(x0 + 1, x1 - tile + 1), step)) or [x0]
        if xs[-1] + tile < x1:
            xs.append(max(x0, x1 - tile))
        ys = list(range(y0, max(y0 + 1, y1 - tile + 1), step)) or [y0]
        if ys[-1] + tile < y1:
            ys.append(max(y0, y1 - tile))
        for ty in ys:
            for tx in xs:
                crop = frame[ty: min(H, ty + tile), tx: min(W, tx + tile)]
                if crop.size == 0:
                    continue
                b, s = self._infer(crop)
                if len(b):
                    b[:, [0, 2]] += tx
                    b[:, [1, 3]] += ty
                    boxes.append(b)
                    scores.append(s)
        if not boxes:
            return []
        B = np.concatenate(boxes)
        S = np.concatenate(scores)
        keep = cv2.dnn.NMSBoxes([[float(a), float(b_), float(c - a), float(d - b_)] for a, b_, c, d in B],
                                S.tolist(), self.conf, self.nms)
        keep = np.asarray(keep).reshape(-1)
        return [Detection(B[i].copy(), float(S[i])) for i in keep]


def filter_players(dets: list[Detection], fmask: np.ndarray, pitch_poly_img: np.ndarray | None = None,
                   max_height_ratio: float = 2.2, frame: np.ndarray | None = None,
                   min_grass_under_foot: float = 0.35) -> tuple[list[Detection], dict]:
    """Drop non-player detections. Returns kept detections and counts per rejection reason.

    Checks (documented in docs/TECHNICAL_REPORT.md):
      1. foot point must lie on the grass mask (rejects photographers/stewards on the track, crowd), and, when the
         frame is given, >= 35% of raw (un-smoothed) grass pixels in a small window just below the foot: the
         smoothed mask can leak over the track next to the boards, the raw pixels under a photographer's feet cannot;
      2. foot point inside the calibrated pitch polygon plus 3 m margin, when calibration exists;
      3. box aspect ratio h/w in [0.9, 6] (rejects partial heads, banners);
      4. height <= max_height_ratio x median on-field height (rejects near-camera spectators that
         overlap the grass in the image).
    """
    H, W = fmask.shape[:2]
    reasons = {"off_grass": 0, "no_grass_under_foot": 0, "outside_pitch": 0, "aspect": 0, "too_large": 0}
    stage = []
    grass = cv2.dilate(fmask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    raw = None
    if frame is not None:
        from .field import GRASS_HI, GRASS_LO
        raw = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), GRASS_LO, GRASS_HI)
    for d in dets:
        fx, fy = d.foot
        xi, yi = int(np.clip(fx, 0, W - 1)), int(np.clip(fy - 2, 0, H - 1))
        if grass[yi, xi] == 0:
            reasons["off_grass"] += 1
            continue
        if raw is not None:
            bw2 = max(3, int(0.35 * (d.box[2] - d.box[0])))
            y0, y1 = int(np.clip(fy, 0, H - 1)), int(np.clip(fy + max(4, 0.15 * d.height), 1, H))
            win = raw[y0:y1, max(0, xi - bw2): min(W, xi + bw2 + 1)]
            if win.size and np.count_nonzero(win) / win.size < min_grass_under_foot:
                reasons["no_grass_under_foot"] += 1
                continue
        if pitch_poly_img is not None and cv2.pointPolygonTest(
                pitch_poly_img.astype(np.float32).reshape(-1, 1, 2), (float(fx), float(fy)), False) < 0:
            reasons["outside_pitch"] += 1
            continue
        bw = d.box[2] - d.box[0]
        if bw <= 0 or not (0.9 <= d.height / bw <= 6.0):
            reasons["aspect"] += 1
            continue
        stage.append(d)
    if stage:
        med = float(np.median([d.height for d in stage]))
        kept = [d for d in stage if d.height <= max_height_ratio * med]
        reasons["too_large"] = len(stage) - len(kept)
    else:
        kept = []
    return kept, reasons
