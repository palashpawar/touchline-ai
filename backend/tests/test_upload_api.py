import importlib

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


def make_video(path, seconds, fps=10, size=(640, 360)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert w.isOpened()
    for i in range(int(seconds * fps)):
        img = np.full((size[1], size[0], 3), (40, 120, 40), np.uint8)
        cv2.circle(img, (20 + 5 * i % (size[0] - 40), size[1] // 2), 8, (255, 255, 255), -1)
        w.write(img)
    w.release()
    return path


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TOUCHLINE_DATA_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("TOUCHLINE_WORKER_MODE", "external")
    monkeypatch.setenv("TOUCHLINE_API_TOKEN", "s3cret")
    monkeypatch.setenv("TOUCHLINE_MAX_UPLOAD_MB", "5")
    monkeypatch.setenv("TOUCHLINE_FRONTEND_DIST", str(tmp_path / "nodist"))
    import touchline.config
    import touchline.api
    importlib.reload(touchline.config)
    api = importlib.reload(touchline.api)
    with TestClient(api.app) as c:
        c.headers["X-Touchline-Token"] = "s3cret"
        yield c


def test_auth_required(client):
    r = client.get("/api/jobs", headers={"X-Touchline-Token": "wrong"})
    assert r.status_code == 401
    assert client.get("/api/health", headers={"X-Touchline-Token": ""}).status_code == 200


def test_valid_upload_creates_job(client, tmp_path):
    p = make_video(tmp_path / "ok.mp4", 12)
    r = client.post("/api/jobs", files={"file": ("ok.mp4", p.read_bytes(), "video/mp4")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["status"] == "uploaded" and j["meta"]["frame_count"] == 120 and "input_key" not in j


@pytest.mark.parametrize("seconds,size,needle", [(3, (640, 360), "between"), (40, (640, 360), "between"),
                                                 (12, (320, 180), "Resolution")])
def test_invalid_videos_rejected(client, tmp_path, seconds, size, needle):
    p = make_video(tmp_path / "bad.mp4", seconds, size=size)
    r = client.post("/api/jobs", files={"file": ("bad.mp4", p.read_bytes(), "video/mp4")})
    assert r.status_code == 400 and needle in r.json()["detail"]


def test_wrong_extension_and_garbage_rejected(client):
    r = client.post("/api/jobs", files={"file": ("clip.exe", b"MZ....", "application/octet-stream")})
    assert r.status_code == 400 and "Unsupported" in r.json()["detail"]
    r = client.post("/api/jobs", files={"file": ("clip.mp4", b"not a video at all" * 100, "video/mp4")})
    assert r.status_code == 400


def test_oversize_upload_rejected(client):
    r = client.post("/api/jobs", files={"file": ("big.mp4", b"\0" * (6 * 1024 * 1024), "video/mp4")})
    assert r.status_code == 413


def test_analyze_requires_valid_calibration(client, tmp_path):
    p = make_video(tmp_path / "ok.mp4", 12)
    jid = client.post("/api/jobs", files={"file": ("ok.mp4", p.read_bytes(), "video/mp4")}).json()["id"]
    assert client.post(f"/api/jobs/{jid}/analyze", json={}).status_code == 400
    bad = client.post(f"/api/jobs/{jid}/calibration", json={
        "frame_index": 0, "image_points": [[10, 10], [20, 10], [30, 10], [40, 10]],
        "rectangle": {"x0": 0, "y0": 13.84, "width_m": 16.5, "height_m": 40.32}}).json()
    assert bad["valid"] is False and bad["errors"]
    good = client.post(f"/api/jobs/{jid}/calibration", json={
        "frame_index": 0, "image_points": [[200, 150], [440, 155], [600, 330], [60, 320]],
        "rectangle": {"x0": 0, "y0": 13.84, "width_m": 16.5, "height_m": 40.32}}).json()
    assert good["valid"], good["errors"]
    r = client.post(f"/api/jobs/{jid}/analyze", json={"initial_stride": 6})
    assert r.status_code == 200 and r.json()["status"] == "queued"
    assert client.post(f"/api/jobs/{jid}/analyze", json={}).status_code == 409  # no double-queue


def test_artifact_names_whitelisted(client, tmp_path):
    p = make_video(tmp_path / "ok.mp4", 12)
    jid = client.post("/api/jobs", files={"file": ("ok.mp4", p.read_bytes(), "video/mp4")}).json()["id"]
    assert client.get(f"/api/jobs/{jid}/artifacts/input.mp4").status_code == 400
    assert client.get(f"/api/jobs/{jid}/artifacts/..%2Fstate.json").status_code in (400, 404)
