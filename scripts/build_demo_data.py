"""Build the static data for the GitHub Pages demo from one real, completed run.

    .venv/bin/python scripts/build_demo_data.py frontend/demo-public/demo

The directory must already contain that run's artifacts (report.json, trace.jsonl, annotated_*.webm,
compare_seg*.jpg). This script adds job.json, trace.json, config.json, health.json and frame0.jpg.
"""
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from touchline import quality  # noqa: E402
from touchline.agent.tools import TOOL_SPECS  # noqa: E402
from touchline.jobs import ALLOWED_EXT  # noqa: E402
from touchline.vision import calibration as cal  # noqa: E402
from touchline.vision.video import read_frame  # noqa: E402

out = Path(sys.argv[1])
report = json.loads((out / "report.json").read_text())
steps = [json.loads(line) for line in (out / "trace.jsonl").read_text().splitlines() if line.strip()]
for s in steps:
    s["run_no"] = report["run_no"]
(out / "trace.json").write_text(json.dumps(steps))

job = dict(report["job"], status="complete", stage="complete", progress=1.0, calibration=report["calibration"],
           error=None, run_no=report["run_no"], active_tasks=0)
(out / "job.json").write_text(json.dumps(job))
(out / "config.json").write_text(json.dumps({
    "limits": {"max_upload_mb": 200, "min_duration_s": 10, "max_duration_s": 30, "allowed_ext": sorted(ALLOWED_EXT)},
    "presets": cal.PRESET_RECTANGLES, "landmarks": cal.standard_landmarks(),
    "signals": quality.spec_table(), "tools": TOOL_SPECS}))
prov = report["provenance"]
(out / "health.json").write_text(json.dumps({"status": "ok", "opencv": prov["opencv"], "storage": "demo",
                                             "processed_on": prov["platform"], "processing_seconds": prov["processing_seconds"]}))
clip = ROOT / "data/clips/france_germany_2018_phase1_ccbysa.ogv"
frame = read_frame(clip, int(report["calibration"].get("frame_index", 0)))
cv2.imwrite(str(out / "frame0.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
print("demo data:", sorted(p.name for p in out.iterdir()))
