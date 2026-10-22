"""Video decoding, metadata and frame sampling (OpenCV VideoCapture)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    frame_count: int  # counted by decoding; container headers are often wrong
    duration_s: float
    codec: str
    fps_source: str = "header"  # "header" or "timestamps" when the container header disagrees with frame timestamps

    def to_dict(self) -> dict:
        return asdict(self)


class VideoError(ValueError):
    pass


def probe(path: Path | str) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError("OpenCV could not open the file as video. Try an MP4 (H.264) or WebM file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip("\x00") or "unknown"
    n = 0
    ts = []
    while cap.grab():
        n += 1
        ts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    cap.release()
    if n == 0 or w == 0 or h == 0:
        raise VideoError("The file contains no decodable video frames.")
    source = "header"
    # Container headers lie (e.g. WebM exported at a 120 fps timebase). Trust decoded timestamps when they disagree.
    if n > 10 and ts[-1] > ts[0]:
        ts_fps = (n - 1) / ((ts[-1] - ts[0]) / 1000.0)
        if not fps or abs(ts_fps - fps) / max(fps, 1e-6) > 0.10:
            fps, source = ts_fps, "timestamps"
    if not (1.0 <= fps <= 240.0):
        raise VideoError(f"Implausible frame rate ({fps:.2f} fps).")
    return VideoMeta(fps=float(fps), width=w, height=h, frame_count=n, duration_s=n / fps, codec=codec,
                     fps_source=source)


def read_frames(path: Path | str, indices: list[int]) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (index, BGR frame) for the requested sorted frame indices, decoding sequentially."""
    wanted = sorted(set(indices))
    if not wanted:
        return
    cap = cv2.VideoCapture(str(path))
    i = 0
    j = 0
    while j < len(wanted):
        if i < wanted[j]:
            if not cap.grab():
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        yield i, frame
        i += 1
        j += 1
    cap.release()


def read_frame(path: Path | str, index: int) -> np.ndarray:
    for _, f in read_frames(path, [index]):
        return f
    raise VideoError(f"frame {index} not available")


def sample_indices(start: int, end: int, stride: int) -> list[int]:
    """Frame indices in [start, end) at a fixed stride; always includes the last frame of the segment."""
    stride = max(1, int(stride))
    idx = list(range(start, end, stride))
    if idx and idx[-1] != end - 1 and end - 1 > idx[-1]:
        idx.append(end - 1)
    return idx
