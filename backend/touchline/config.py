"""Runtime settings, read once from environment variables (see docs/ENVIRONMENT.md)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    storage_backend: str = field(default_factory=lambda: _env("TOUCHLINE_STORAGE", "local"))  # local | s3
    data_dir: Path = field(default_factory=lambda: Path(_env("TOUCHLINE_DATA_DIR", str(ROOT / "var"))))
    s3_bucket: str = field(default_factory=lambda: _env("TOUCHLINE_S3_BUCKET", ""))
    s3_prefix: str = field(default_factory=lambda: _env("TOUCHLINE_S3_PREFIX", "touchline/"))
    # Only for local testing against an S3 emulator; leave unset on AWS.
    s3_endpoint: str = field(default_factory=lambda: _env("TOUCHLINE_S3_ENDPOINT", ""))
    aws_region: str = field(default_factory=lambda: _env("AWS_REGION", "us-east-1"))
    model_path: Path = field(default_factory=lambda: Path(_env(
        "TOUCHLINE_MODEL_PATH", str(ROOT / "models" / "object_detection_yolox_2022nov.onnx"))))
    max_upload_mb: int = field(default_factory=lambda: int(_env("TOUCHLINE_MAX_UPLOAD_MB", "200")))
    min_duration_s: float = field(default_factory=lambda: float(_env("TOUCHLINE_MIN_DURATION_S", "10")))
    max_duration_s: float = field(default_factory=lambda: float(_env("TOUCHLINE_MAX_DURATION_S", "30")))
    worker_concurrency: int = field(default_factory=lambda: int(_env("TOUCHLINE_WORKER_CONCURRENCY", "1")))
    # When set, every /api request must send header "X-Touchline-Token: <value>".
    api_token: str = field(default_factory=lambda: _env("TOUCHLINE_API_TOKEN", ""))
    # "inline": the API process also runs the worker. "external": a separate worker process polls SQLite.
    worker_mode: str = field(default_factory=lambda: _env("TOUCHLINE_WORKER_MODE", "inline"))
    frontend_dist: Path = field(default_factory=lambda: Path(_env(
        "TOUCHLINE_FRONTEND_DIST", str(ROOT / "frontend" / "dist"))))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "touchline.sqlite3"


def get_settings() -> Settings:
    return Settings()
