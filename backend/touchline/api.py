"""FastAPI service: upload, calibration, analysis queueing, live agent trace, results, review actions."""
from __future__ import annotations

import hmac
import re
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import jobs, quality
from .agent.tools import TOOL_SPECS
from .config import get_settings
from .db import DB
from .logging_setup import setup
from .storage import make_storage
from .vision import calibration as cal
from .worker import Worker

settings = get_settings()
setup()
db = DB(settings.db_path)
storage = make_storage(settings)
worker = Worker(db, storage, settings)
ARTIFACT_RE = re.compile(r"^(annotated_(final|initial)\.(webm|mp4)|compare_seg\d+\.jpg|report\.json|trace\.jsonl)$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.worker_mode == "inline":
        worker.start()
    yield
    worker.stop.set()


app = FastAPI(title="Touchline AI", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def auth(request: Request, call_next):
    path = request.url.path
    if settings.api_token and path.startswith("/api/") and path not in ("/api/health", "/api/auth"):
        tok = request.headers.get("x-touchline-token") or request.cookies.get("touchline_token") or ""
        if not hmac.compare_digest(tok, settings.api_token):
            return JSONResponse({"detail": "Authentication required: enter the access token."}, status_code=401)
    return await call_next(request)


class AuthBody(BaseModel):
    token: str


@app.post("/api/auth")
def login(body: AuthBody, request: Request):
    if not settings.api_token:
        return {"ok": True, "auth": "disabled"}
    if not hmac.compare_digest(body.token, settings.api_token):
        raise HTTPException(401, "Invalid token")
    resp = JSONResponse({"ok": True})
    resp.set_cookie("touchline_token", body.token, httponly=True, samesite="strict",
                    secure=request.url.scheme == "https", max_age=8 * 3600)
    return resp


@app.get("/api/health")
def health():
    ok_model = settings.model_path.exists()
    return {"status": "ok" if ok_model else "degraded", "opencv": cv2.__version__, "storage": settings.storage_backend,
            "model_present": ok_model, "worker_mode": settings.worker_mode,
            "worker_concurrency": settings.worker_concurrency, "auth": bool(settings.api_token)}


@app.get("/api/config")
def config():
    return {"limits": {"max_upload_mb": settings.max_upload_mb, "min_duration_s": settings.min_duration_s,
                       "max_duration_s": settings.max_duration_s, "allowed_ext": sorted(jobs.ALLOWED_EXT)},
            "presets": cal.PRESET_RECTANGLES, "landmarks": cal.standard_landmarks(),
            "signals": quality.spec_table(), "tools": TOOL_SPECS}


def _job(jid: str) -> dict:
    j = db.get_job(jid)
    if not j:
        raise HTTPException(404, "Job not found")
    return j


def _public(j: dict) -> dict:
    j = dict(j)
    j.pop("input_key", None)
    j["active_tasks"] = len(db.active_tasks(j["id"]))
    return j


@app.get("/api/jobs")
def list_jobs():
    return [_public(j) for j in db.list_jobs()]


@app.post("/api/jobs")
async def upload(request: Request, file: UploadFile = File(...)):
    cl = request.headers.get("content-length")
    if cl and int(cl) > (settings.max_upload_mb + 1) * 1024 * 1024:
        raise HTTPException(413, f"File exceeds the {settings.max_upload_mb} MB upload limit.")
    try:
        jobs.validate_upload(file.filename or "", 1, settings)
    except jobs.UploadError as e:
        raise HTTPException(400, str(e))
    tmpdir = Path(tempfile.mkdtemp(prefix="upload-"))
    dst = tmpdir / ("clip" + Path(file.filename or "x.mp4").suffix.lower())
    size, limit = 0, settings.max_upload_mb * 1024 * 1024
    try:
        with dst.open("wb") as fh:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"File exceeds the {settings.max_upload_mb} MB upload limit.")
                fh.write(chunk)
        try:
            job = jobs.ingest(db, storage, settings, file.filename or dst.name, dst)
        except jobs.UploadError as e:
            raise HTTPException(400, str(e))
        return _public(job)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    return _public(_job(jid))


@app.get("/api/jobs/{jid}/frame/{index}")
def frame(jid: str, index: int):
    j = _job(jid)
    if not 0 <= index < j["meta"]["frame_count"]:
        raise HTTPException(400, "frame index out of range")
    return Response(jobs.frame_jpeg(storage, j, index), media_type="image/jpeg",
                    headers={"Cache-Control": "private, max-age=3600"})


@app.get("/api/jobs/{jid}/lines/{index}")
def lines(jid: str, index: int):
    return {"segments": jobs.line_segments(storage, _job(jid), index)}


class CalibrationBody(BaseModel):
    frame_index: int = 0
    image_points: list[list[float]] = Field(..., min_length=4, max_length=4)
    pitch_points: list[list[float]] | None = None
    rectangle: dict | None = None
    pitch_size: list[float] | None = None
    holdout: list[dict] | None = None
    label: str | None = None


@app.post("/api/jobs/{jid}/calibration")
def calibration(jid: str, body: CalibrationBody):
    j = _job(jid)
    if db.active_tasks(jid):
        raise HTTPException(409, "Analysis is running; wait for it to finish before recalibrating.")
    if body.pitch_points is None and body.rectangle is None:
        raise HTTPException(400, "Provide either pitch_points or rectangle")
    try:
        return jobs.calibrate(db, storage, j, body.model_dump())
    except jobs.UploadError as e:
        raise HTTPException(400, str(e))


class AnalyzeBody(BaseModel):
    initial_stride: int = Field(6, ge=1, le=12)
    team_colors: list[str] | None = None


@app.post("/api/jobs/{jid}/analyze")
def analyze(jid: str, body: AnalyzeBody):
    j = _job(jid)
    if db.active_tasks(jid):
        raise HTTPException(409, "This job already has a queued or running task.")
    if not (j.get("calibration") or {}).get("valid"):
        raise HTTPException(400, "Submit a valid calibration before analysing; metric distances require it.")
    if body.team_colors and not all(re.fullmatch(r"#[0-9a-fA-F]{6}", c) for c in body.team_colors):
        raise HTTPException(400, "team_colors must be two #rrggbb values")
    db.enqueue(jid, "analyze", body.model_dump())
    db.update_job(jid, status="queued", stage="queued", progress=0.0)
    return _public(db.get_job(jid))


@app.get("/api/jobs/{jid}/trace")
def trace(jid: str, run_no: int | None = None):
    j = _job(jid)
    return db.steps(jid, run_no if run_no is not None else j["run_no"])


@app.get("/api/jobs/{jid}/reviews")
def reviews(jid: str):
    _job(jid)
    return db.reviews(jid)


class ReviewBody(BaseModel):
    action: str  # approve_rerun | accept_uncertain | team_update
    note: str | None = None
    team_colors: list[str] | None = None
    team_overrides: dict[str, str] | None = None


@app.post("/api/jobs/{jid}/reviews/{rid}")
def review(jid: str, rid: str, body: ReviewBody):
    _job(jid)
    if db.active_tasks(jid):
        raise HTTPException(409, "A task is already running for this job.")
    if body.action not in ("approve_rerun", "accept_uncertain", "team_update"):
        raise HTTPException(400, "Unknown review action")
    if body.action != "team_update":
        rv = db.get_review(rid)
        if not rv or rv["job_id"] != jid:
            raise HTTPException(404, "Review not found")
        if rv["status"] != "open":
            raise HTTPException(409, "Review already resolved")
        if body.action == "approve_rerun" and not rv["proposed_config"]:
            raise HTTPException(400, "This review has no proposed rerun; recalibrate or accept instead.")
    db.enqueue(jid, "review", {"review_id": None if body.action == "team_update" else rid, **body.model_dump()})
    db.update_job(jid, status="queued", stage=f"queued: {body.action}")
    return _public(db.get_job(jid))


@app.get("/api/jobs/{jid}/report")
def report(jid: str):
    j = _job(jid)
    key = f"jobs/{jid}/runs/{j['run_no']}/report.json"
    if not j["run_no"] or not storage.exists(key):
        raise HTTPException(404, "No report yet: run the analysis first.")
    return Response(storage.read_bytes(key), media_type="application/json")


@app.get("/api/jobs/{jid}/artifacts/{name}")
def artifact(jid: str, name: str, download: bool = False):
    j = _job(jid)
    if not ARTIFACT_RE.match(name):
        raise HTTPException(400, "Unknown artifact")
    key = f"jobs/{jid}/runs/{j['run_no']}/{name}"
    try:
        path = storage.fetch(key)
    except Exception:
        raise HTTPException(404, "Artifact not found")
    media = {".webm": "video/webm", ".mp4": "video/mp4", ".jpg": "image/jpeg", ".json": "application/json",
             ".jsonl": "application/x-ndjson"}[Path(name).suffix]
    return FileResponse(path, media_type=media, filename=f"touchline_{jid}_{name}" if download else None)


if settings.frontend_dist.exists():
    app.mount("/", StaticFiles(directory=settings.frontend_dist, html=True), name="ui")
