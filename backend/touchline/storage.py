"""Artifact storage behind one interface: local filesystem (dev) or S3 (AWS).

Keys are relative POSIX paths such as ``jobs/<job_id>/input.mp4``. Processing code
always works on local files; ``fetch`` materialises an object into a local cache and
``put_file`` uploads a finished artifact.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Protocol

from .config import Settings


class Storage(Protocol):
    def put_file(self, key: str, path: Path, content_type: str | None = None) -> None: ...
    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None: ...
    def fetch(self, key: str) -> Path: ...
    def exists(self, key: str) -> bool: ...
    def read_bytes(self, key: str) -> bytes: ...


class LocalStorage:
    def __init__(self, root: Path):
        self.root = root / "objects"
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents:
            raise ValueError(f"invalid storage key {key!r}")
        return p

    def put_file(self, key: str, path: Path, content_type: str | None = None) -> None:
        dst = self._p(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if Path(path).resolve() != dst:
            shutil.copyfile(path, dst)

    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        dst = self._p(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)

    def fetch(self, key: str) -> Path:
        p = self._p(key)
        if not p.exists():
            raise FileNotFoundError(key)
        return p

    def exists(self, key: str) -> bool:
        return self._p(key).exists()

    def read_bytes(self, key: str) -> bytes:
        return self.fetch(key).read_bytes()


class S3Storage:
    """S3-backed storage. Credentials come from the EC2 instance role (never embedded)."""

    def __init__(self, bucket: str, prefix: str, region: str, cache_dir: Path, client=None, endpoint: str = ""):
        import boto3

        if not bucket:
            raise ValueError("TOUCHLINE_S3_BUCKET is required when TOUCHLINE_STORAGE=s3")
        self.bucket = bucket
        self.prefix = prefix if prefix.endswith("/") else prefix + "/"
        self.s3 = client or boto3.client("s3", region_name=region, endpoint_url=endpoint or None)
        self.cache = cache_dir / "s3cache"
        self.cache.mkdir(parents=True, exist_ok=True)

    def _k(self, key: str) -> str:
        if ".." in key.split("/"):
            raise ValueError(f"invalid storage key {key!r}")
        return self.prefix + key

    def _extra(self, content_type: str | None) -> dict:
        extra = {"ServerSideEncryption": "AES256"}
        if content_type:
            extra["ContentType"] = content_type
        return extra

    def put_file(self, key: str, path: Path, content_type: str | None = None) -> None:
        self.s3.upload_file(str(path), self.bucket, self._k(key), ExtraArgs=self._extra(content_type))
        local = self.cache / key
        local.parent.mkdir(parents=True, exist_ok=True)
        if Path(path).resolve() != local.resolve():
            shutil.copyfile(path, local)

    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=self._k(key), Body=data, **self._extra(content_type))
        local = self.cache / key
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)

    def fetch(self, key: str) -> Path:
        local = self.cache / key
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(self.bucket, self._k(key), str(local))
        return local

    def exists(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._k(key))
            return True
        except Exception:
            return False

    def read_bytes(self, key: str) -> bytes:
        return self.fetch(key).read_bytes()


def put_json(storage: Storage, key: str, obj) -> None:
    storage.put_bytes(key, json.dumps(obj, indent=2, default=str).encode(), "application/json")


def make_storage(settings: Settings) -> Storage:
    if settings.storage_backend == "s3":
        return S3Storage(settings.s3_bucket, settings.s3_prefix, settings.aws_region, settings.data_dir,
                         endpoint=settings.s3_endpoint)
    return LocalStorage(settings.data_dir)
