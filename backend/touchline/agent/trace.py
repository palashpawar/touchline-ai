"""Agent trace: one record per agent step, serialisable to JSON lines and back."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

STATES = ("INGEST", "CALIBRATE", "ANALYZE", "ASSESS_QUALITY", "PLAN_RECOVERY", "RERUN_SEGMENT",
          "COMPARE", "REQUEST_HUMAN_REVIEW", "COMPLETE")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class AgentStep:
    run_id: str
    segment_id: str | None
    state: str
    decision: str
    reason: str
    tool: str | None = None
    tool_args: dict = field(default_factory=dict)
    observed_signals: dict = field(default_factory=dict)
    previous_config: dict | None = None
    proposed_config: dict | None = None
    quality_before: dict | None = None
    quality_after: dict | None = None
    outcome: str = ""
    accepted: bool | None = None
    attempt: int = 0
    actor: str = "agent"  # "agent" | "human"
    timestamp: str = field(default_factory=now_iso)
    step: int = 0

    def __post_init__(self):
        if self.state not in STATES:
            raise ValueError(f"unknown agent state {self.state}")

    def to_dict(self) -> dict:
        return asdict(self)


class Trace:
    def __init__(self, run_id: str, sink: Callable[[AgentStep], None] | None = None):
        self.run_id = run_id
        self.steps: list[AgentStep] = []
        self.sink = sink

    def log(self, **kw: Any) -> AgentStep:
        st = AgentStep(run_id=self.run_id, step=len(self.steps) + 1, **kw)
        self.steps.append(st)
        if self.sink:
            self.sink(st)
        return st

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(s.to_dict(), default=_json_default) for s in self.steps) + "\n"

    def to_list(self) -> list[dict]:
        return json.loads(json.dumps([s.to_dict() for s in self.steps], default=_json_default))

    @classmethod
    def from_jsonl(cls, text: str) -> "Trace":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        tr = cls(rows[0]["run_id"] if rows else "")
        for r in rows:
            tr.steps.append(AgentStep(**r))
        return tr


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(o)
