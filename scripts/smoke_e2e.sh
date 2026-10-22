#!/usr/bin/env bash
# End-to-end smoke test over HTTP: upload -> calibrate -> analyze -> poll -> download report/trace/video.
# Usage: scripts/smoke_e2e.sh [BASE_URL] [CLIP] [TOKEN]
set -euo pipefail
BASE=${1:-http://localhost:8780}
CLIP=${2:-data/clips/france_germany_2018_phase1_ccbysa.ogv}
TOKEN=${3:-${TOUCHLINE_API_TOKEN:-}}
OUT=${OUT:-eval/out/smoke}
mkdir -p "$OUT"
H=(-H "X-Touchline-Token: ${TOKEN}")
curl -sf "${H[@]}" "$BASE/api/health" | tee "$OUT/health.json"; echo
JOB=$(curl -sf "${H[@]}" -F "file=@${CLIP}" "$BASE/api/jobs" | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')
echo "job $JOB"
# Demo calibration for the France-Germany clip (frame 0): four near-side penalty/goal-area landmarks.
curl -sf "${H[@]}" -H 'Content-Type: application/json' "$BASE/api/jobs/$JOB/calibration" -d '{
  "frame_index": 0, "label": "near-side box landmarks",
  "image_points": [[520,836],[640,811],[1190,795],[928,873]],
  "pitch_points": [[0,43.16],[5.5,43.16],[16.5,54.16],[0,54.16]],
  "holdout": [{"name":"left_goal_near_post","image":[320,833],"pitch":[0,37.66]},
              {"name":"left_goal_far_post","image":[95,818],"pitch":[0,30.34]}]}' > "$OUT/calibration.json"
python3 -c 'import json;d=json.load(open("'"$OUT"'/calibration.json"));print("calibration valid:",d["valid"],d["errors"],d["warnings"])'
curl -sf "${H[@]}" -H 'Content-Type: application/json' "$BASE/api/jobs/$JOB/analyze" -d '{"initial_stride":6}' > /dev/null
T0=$(date +%s)
while true; do
  S=$(curl -sf "${H[@]}" "$BASE/api/jobs/$JOB" | python3 -c 'import sys,json;j=json.load(sys.stdin);print(j["status"],j["progress"],j["stage"].replace(" ","_"))')
  echo "$(( $(date +%s) - T0 ))s $S"
  case "$S" in complete*|needs_review*|failed*) break;; esac
  sleep 10
done
curl -sf "${H[@]}" "$BASE/api/jobs/$JOB/report" -o "$OUT/report.json"
curl -sf "${H[@]}" "$BASE/api/jobs/$JOB/artifacts/trace.jsonl" -o "$OUT/trace.jsonl"
curl -sf "${H[@]}" "$BASE/api/jobs/$JOB/artifacts/annotated_final.webm" -o "$OUT/annotated_final.webm"
echo "$JOB" > "$OUT/job_id"
ls -la "$OUT"
