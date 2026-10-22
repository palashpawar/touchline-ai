"""SQLite job metadata: jobs, queued tasks, live agent steps, review requests."""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, created_at TEXT, updated_at TEXT, status TEXT, stage TEXT, progress REAL DEFAULT 0,
  filename TEXT, input_key TEXT, meta TEXT, calibration TEXT, options TEXT, error TEXT, run_no INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS tasks (
  id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, kind TEXT, payload TEXT, status TEXT DEFAULT 'queued',
  created_at TEXT, started_at TEXT, finished_at TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS steps (
  id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, run_no INTEGER, body TEXT
);
CREATE TABLE IF NOT EXISTS reviews (
  id TEXT PRIMARY KEY, job_id TEXT, run_no INTEGER, segment_id TEXT, kind TEXT, reason TEXT,
  proposed_config TEXT, status TEXT DEFAULT 'open', resolution TEXT, created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_steps_job ON steps(job_id, run_no);
"""

JSON_COLS = {"meta", "calibration", "options", "payload", "body", "proposed_config", "resolution"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        try:
            yield c
            c.commit()
        finally:
            c.close()

    @staticmethod
    def _row(r) -> dict | None:
        if r is None:
            return None
        d = dict(r)
        for k in JSON_COLS & d.keys():
            if d[k] is not None:
                d[k] = json.loads(d[k])
        return d

    # jobs
    def create_job(self, filename: str, input_key: str, meta: dict, job_id: str | None = None) -> dict:
        jid = job_id or uuid.uuid4().hex[:12]
        with self.conn() as c:
            c.execute("INSERT INTO jobs(id,created_at,updated_at,status,stage,filename,input_key,meta,options) "
                      "VALUES(?,?,?,?,?,?,?,?,?)", (jid, now(), now(), "uploaded", "awaiting calibration", filename,
                                                    input_key, json.dumps(meta), json.dumps({})))
        return self.get_job(jid)

    def get_job(self, jid: str) -> dict | None:
        with self.conn() as c:
            return self._row(c.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone())

    def list_jobs(self, limit: int = 20) -> list[dict]:
        with self.conn() as c:
            return [self._row(r) for r in c.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))]

    def update_job(self, jid: str, **fields) -> None:
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields) + ", updated_at=?"
        vals = [json.dumps(v) if k in JSON_COLS and v is not None else v for k, v in fields.items()]
        with self.conn() as c:
            c.execute(f"UPDATE jobs SET {sets} WHERE id=?", (*vals, now(), jid))

    # tasks
    def enqueue(self, job_id: str, kind: str, payload: dict) -> int:
        with self.conn() as c:
            cur = c.execute("INSERT INTO tasks(job_id,kind,payload,created_at) VALUES(?,?,?,?)",
                            (job_id, kind, json.dumps(payload), now()))
            return int(cur.lastrowid)

    def claim_task(self) -> dict | None:
        with self._lock, self.conn() as c:
            c.execute("BEGIN IMMEDIATE")
            r = c.execute("SELECT * FROM tasks WHERE status='queued' ORDER BY id LIMIT 1").fetchone()
            if r is None:
                return None
            c.execute("UPDATE tasks SET status='running', started_at=? WHERE id=?", (now(), r["id"]))
            return self._row(r)

    def finish_task(self, tid: int, error: str | None = None) -> None:
        with self.conn() as c:
            c.execute("UPDATE tasks SET status=?, finished_at=?, error=? WHERE id=?",
                      ("failed" if error else "done", now(), error, tid))

    def active_tasks(self, job_id: str) -> list[dict]:
        with self.conn() as c:
            return [self._row(r) for r in c.execute(
                "SELECT * FROM tasks WHERE job_id=? AND status IN ('queued','running')", (job_id,))]

    def requeue_stale(self) -> int:
        """Tasks left 'running' by a crashed worker are failed explicitly (never silently retried)."""
        with self.conn() as c:
            return c.execute("UPDATE tasks SET status='failed', error='worker restarted' WHERE status='running'").rowcount

    # agent steps
    def add_step(self, job_id: str, run_no: int, body: dict) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO steps(job_id,run_no,body) VALUES(?,?,?)", (job_id, run_no, json.dumps(body, default=str)))

    def steps(self, job_id: str, run_no: int | None = None) -> list[dict]:
        q, args = "SELECT * FROM steps WHERE job_id=?", [job_id]
        if run_no is not None:
            q += " AND run_no=?"
            args.append(run_no)
        with self.conn() as c:
            return [dict(self._row(r)["body"], run_no=r["run_no"]) for r in c.execute(q + " ORDER BY id", args)]

    # reviews
    def add_review(self, job_id: str, run_no: int, rv: dict) -> None:
        with self.conn() as c:
            c.execute("INSERT OR IGNORE INTO reviews(id,job_id,run_no,segment_id,kind,reason,proposed_config,status,created_at) "
                      "VALUES(?,?,?,?,?,?,?,?,?)", (f"{job_id}-r{run_no}-{rv['review_id']}", job_id, run_no, rv["segment_id"],
                                                    rv["kind"], rv["reason"], json.dumps(rv.get("proposed_config")), "open", now()))

    def reviews(self, job_id: str) -> list[dict]:
        with self.conn() as c:
            return [self._row(r) for r in c.execute("SELECT * FROM reviews WHERE job_id=? ORDER BY created_at, id", (job_id,))]

    def get_review(self, rid: str) -> dict | None:
        with self.conn() as c:
            return self._row(c.execute("SELECT * FROM reviews WHERE id=?", (rid,)).fetchone())

    def resolve_review(self, rid: str, status: str, resolution: dict) -> None:
        with self.conn() as c:
            c.execute("UPDATE reviews SET status=?, resolution=? WHERE id=?", (status, json.dumps(resolution), rid))
