"""Background worker with bounded concurrency.

Tasks are rows in SQLite. N worker threads (TOUCHLINE_WORKER_CONCURRENCY, default 1) claim tasks
atomically, so at most N clips are processed at once regardless of how many are queued.
Run standalone with ``python -m touchline.worker`` (Docker "worker" service) or inline in the API.
"""
from __future__ import annotations

import logging
import signal
import threading
import time
import traceback

from . import jobs
from .config import Settings, get_settings
from .db import DB
from .logging_setup import setup
from .storage import Storage, make_storage

log = logging.getLogger("touchline.worker")


class Worker:
    def __init__(self, db: DB, storage: Storage, settings: Settings):
        self.db, self.storage, self.settings = db, storage, settings
        self.stop = threading.Event()
        self.threads: list[threading.Thread] = []

    def handle(self, task: dict) -> None:
        jid = task["job_id"]
        if task["kind"] == "analyze":
            jobs.run_analysis(self.db, self.storage, self.settings, jid, task["payload"])
        elif task["kind"] == "review":
            jobs.review_action(self.db, self.storage, self.settings, jid, task["payload"])
        else:
            raise ValueError(f"unknown task kind {task['kind']}")

    def loop(self) -> None:
        while not self.stop.is_set():
            task = self.db.claim_task()
            if task is None:
                self.stop.wait(1.0)
                continue
            t0 = time.time()
            log.info("task started", extra={"job_id": task["job_id"], "task_id": task["id"], "event": task["kind"]})
            try:
                self.handle(task)
                self.db.finish_task(task["id"])
                log.info("task done", extra={"job_id": task["job_id"], "task_id": task["id"],
                                             "elapsed_s": round(time.time() - t0, 1)})
            except Exception as e:  # report the failure on the job; never crash the worker
                log.error("task failed", exc_info=True, extra={"job_id": task["job_id"], "task_id": task["id"]})
                self.db.finish_task(task["id"], error=f"{type(e).__name__}: {e}")
                self.db.update_job(task["job_id"], status="failed", stage="processing failed",
                                   error=f"{type(e).__name__}: {e}\n" + traceback.format_exc(limit=3))

    def start(self) -> None:
        n = self.db.requeue_stale()
        if n:
            log.warning("marked stale running tasks failed", extra={"event": "stale_tasks", "status": n})
        for i in range(max(1, self.settings.worker_concurrency)):
            t = threading.Thread(target=self.loop, name=f"worker-{i}", daemon=True)
            t.start()
            self.threads.append(t)


def main() -> None:
    setup()
    settings = get_settings()
    w = Worker(DB(settings.db_path), make_storage(settings), settings)
    w.start()
    signal.signal(signal.SIGTERM, lambda *_: w.stop.set())
    signal.signal(signal.SIGINT, lambda *_: w.stop.set())
    log.info("worker running", extra={"event": "worker_start", "status": settings.worker_concurrency})
    while not w.stop.is_set():
        time.sleep(0.5)


if __name__ == "__main__":
    main()
