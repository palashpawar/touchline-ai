# Touchline AI — technical report

## 1. Problem and users

College soccer staff often have only phone or stand footage. Tools that turn that footage into "team width" or
heatmaps tend to report numbers without saying whether the geometry behind them still holds. In a 17 s handheld clip, the
camera in our demo drifts **135 px** away from the frame where the pitch was calibrated. A static calibration then quietly
puts players several metres away from where they really are.

Touchline AI is for **coaches and student analysts** reviewing one short clip at a time. Its defining feature is an agent
that measures the reliability of its own visual analysis, changes the processing approach when the evidence says the
first pass is unreliable, reruns only the affected segment, and keeps a correction only if it validates. If it cannot
recover, it asks a human.

## 2. Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for Mermaid diagrams of the system, the vision pipeline, and the agent state machine.

* **API** (FastAPI): upload validation, which decodes every frame to check duration, resolution and readability; streaming size
  limit; calibration; task queueing; live trace; artifacts; optional token auth (header or HttpOnly cookie).
* **Worker:** N threads (default 1) claim tasks atomically from SQLite, so at most N clips are processed at once. It runs as a separate
  container (`python -m touchline.worker`) or in-process for development.
* **Storage:** one `Storage` protocol with local-filesystem and S3 implementations. S3 writes use SSE-S3, keys sit under a
  prefix, and credentials come from the instance role.
* **Frontend** (React/TS/Vite): upload → calibrate → analyze (live timeline) → results/review/export.

## 3. OpenCV 5 operations

The runtime is the official `opencv-python-headless==5.0.0.93` wheel (OpenCV 5.0.0, released 2026-06-06). The Docker build
asserts `cv2.__version__.startswith("5.")`, and every report records the version. One OpenCV 5 API change surfaced during
development: `HoughLinesP` returns `(N, 4)` instead of `(N, 1, 4)`. The code normalizes the shape (`vision/field.py`).

| Stage | OpenCV functions |
|---|---|
| Decode, metadata, sampling | `VideoCapture`, `grab`/`read`, `CAP_PROP_FPS`/`POS_MSEC` (timestamp-derived fps when the header disagrees) |
| Field segmentation | `cvtColor(HSV)`, `inRange`, `morphologyEx(CLOSE)`, `connectedComponentsWithStats` |
| Line detection / calibration aid | `morphologyEx(TOPHAT)`, `HoughLinesP` (snap-to-corner in the UI) |
| Calibration | `findHomography` (4-point exact), `perspectiveTransform`, `pointPolygonTest` |
| Detection | `dnn.readNet`, `Net.forward` (YOLOX-S ONNX), `resize`, `dnn.NMSBoxes` |
| Camera motion | `ORB_create`, `BFMatcher.knnMatch` (ratio test), `goodFeaturesToTrack`, `calcOpticalFlowPyrLK` (forward–backward check), `findHomography(USAC_MAGSAC)` |
| Calibration validation after motion | `warpPerspective` of the line mask, `distanceTransform` (line-alignment signal) |
| Tracking | `KalmanFilter` (constant velocity), `calcHist` + `compareHist` (Bhattacharyya) appearance term |
| Teams | `cvtColor(Lab)`, `kmeans` (k-means++) |
| Output | `rectangle`/`polylines`/`putText`, pitch template via inverse homography, `VideoWriter` (VP8/WebM), `imencode` |

The YOLOX-S weights are an external pretrained detector (OpenCV Zoo, Apache-2.0). OpenCV provides the inference runtime,
not the learned model. Association uses SciPy's Hungarian solver.

## 4. Calibration

The user maps four image points to a known pitch region, in a fixed cyclic order P1→P4. The region can be a marked rectangle
(penalty area, goal area), a custom rectangle with entered dimensions, or four named landmarks. The tool never assumes all four
pitch corners are visible. `validate_calibration` rejects:

* fewer or more than 4 points, non-finite values, points more than 5% outside the frame;
* points closer than 12 px, a quad area below 0.2% of the frame, or any three points nearly collinear (triangle/quad area below 4%);
* crossed edges (bow-tie) or a non-convex quad, which mean the points are out of order;
* a **mirrored** order: the image and pitch quads have opposite orientation. The pitch frame shares image handedness, so a real
  camera cannot produce this;
* a horizon through the calibration quad, an implied scale below 0.3 px/m, or extreme anisotropy (warning).

It also reports the visible-pitch fraction and extent, obtained by projecting the grass mask. If less than 60% of the pitch
width is visible, team width is labelled a partial-view measurement. It reports held-out landmark error in metres when check
points are given. Metric outputs (positions, trails, heatmap, width/depth) are **withheld entirely** when calibration is
invalid.

**Camera movement:** each sampled frame is matched to the calibration frame. When the agent decides to compensate,
the per-frame image→pitch homography becomes `H_ref · M_t`, where `M_t` maps frame t to the reference frame. The update is
accepted only if the motion model validates *and* the pitch-line alignment on probe frames improves and exceeds 0.6.
Otherwise the segment is marked unreliable and recalibration is requested.

## 5. Detection, filtering, tracking

YOLOX-S runs on overlapping 640 px tiles that cover the grass bounding box. Players are 30–60 px tall in a 1080p wide shot,
too small for a single 640 px pass. The ground-contact point is the box bottom-centre. Non-players are removed when:
(1) the foot is not on the grass mask (photographers and stewards on the track, the crowd); (2) the foot projects more than 3 m outside the pitch;
(3) the aspect ratio is implausible; (4) the box is more than 2.2× the median on-field height (near-camera spectators).

The tracker is a constant-velocity Kalman filter per track in reference-frame image coordinates, with Hungarian assignment on
position residual (in box heights) plus a jersey-histogram term. New tracks are tentative until their second hit. Unmatched
tracks coast for up to 1.0 s to bridge occlusion. Tracks are stitched across segment boundaries by matching adjacent-frame
foot points. **There is no re-identification:** a player who is occluded for more than 1 s, or who leaves the view, returns
with a new ID, and crossing players can swap IDs.

## 6. Agent behaviour

The agent is **rule-based**: a deterministic policy over measured signals (`agent/policy.py`), described as such everywhere.
There is no LLM in the control path, and no paid API is required.

Each ~4 s segment runs: ANALYZE (initial schedule: every 6th frame, static calibration) → ASSESS_QUALITY → PLAN_RECOVERY.
The plan is a function of the breached signals:

| Evidence (breached signals) | Precondition | Action (tool calls) |
|---|---|---|
| `calibration_valid` | — | `request_human_review(recalibrate)` |
| `camera_drift_px` or `line_alignment` | not yet compensated, motion model validated | `estimate_camera_motion` → `update_calibration` → `rerun_segment(motion_compensation)` |
| same | motion model **not** validated | `request_human_review(recalibrate)` |
| `line_alignment` | already compensated | `request_human_review(recalibrate)` |
| `short_track_ratio`, `fragmentation_index`, `assoc_residual_h`, `implausible_jump_rate`, `detection_gap_fraction` | stride > 1 | `rerun_segment(stride 6→3→2→1)` |
| any | 2 attempts used | `request_human_review(approve_rerun, proposed config)` |

`compare_runs` evaluates both runs on **identical frames**. Every stride divides 6, so the original frames are always present.
A correction is accepted only if a target signal improves by at least 10% and no target or guard signal degrades
materially. Otherwise the previous result is preserved, the segment is marked `uncertain`, and a review is opened. Signal
definitions, units, thresholds and limitations are in [QUALITY_SIGNALS.md](QUALITY_SIGNALS.md).

Every step is logged with timestamp, run and segment IDs, observed signals, decision and reason, tool name and arguments,
previous and proposed config, quality before and after, outcome, and acceptance. The trace is available live in the UI and exported as
JSONL. [docs/demo/trace.jsonl](demo/trace.jsonl) comes from a real run.

**Why it qualifies as perception → decision → action:** the tool called next, and its arguments, are determined by OpenCV
measurements of that specific segment. On the control clip the agent calls nothing after analysis; on drift it calls the
motion tools; on fragmentation it changes the sampling stride. `tests/test_agent_loop.py::test_decision_depends_on_signals_not_fixed_sequence`
asserts this.

## 7. Evaluation

All numbers come from `eval/run_eval.py` (full output: [eval/RESULTS.md](../eval/RESULTS.md), `eval/out/results.json`).
The hardware was an Apple M3 CPU (8 cores, no GPU), macOS, OpenCV 5.0.0, and the whole harness took about 12 minutes.

**Disclosure:** two design changes were made *after* seeing a first evaluation round.
(1) The off-pitch signal now counts only detections well inside the grass. A spectator-filter leak had made a static
control clip escalate. The filter itself was also fixed to require raw grass pixels under the foot.
(2) The camera-geometry correction is now judged on geometric signals only. Jump rate inflated by sped-up footage had
vetoed corrections that validated. The 5x sampling-stress case was added after the 3x case's result. The tables below
are from the re-run with the final code.

**Agent trigger accuracy: 8/9 cases.**

| Case | Kind | Expected | Observed |
|---|---|---|---|
| natural | natural | camera-motion update | all 4 segments: camera-motion update, accepted |
| control_static | synthetic | no action | accept ×3, no recovery calls |
| synthetic_pan | synthetic | camera-motion update | 3/3 segments updated and accepted |
| textureless_pan | synthetic | human review | update attempted, rejected at COMPARE (alignment 0.17→0.18) → review |
| scene_cut | synthetic edit of real footage | fix before cut, escalate after | seg1 updated; post-cut segments → review |
| time_compressed (3x) | synthetic | densify sampling | densify triggered; accepted in seg2, rejected (9% < 10%) in seg1 → review |
| time_compressed_5x | synthetic | densify sampling | camera update accepted, then densify accepted, then retry limit → review |
| swapped_points | synthetic | calibration rejected | rejected at CALIBRATE ("edges cross") |
| shifted_points | synthetic | human review | **missed**: geometry was consistently wrong, and line alignment cannot see that |

**Did recoveries improve the chosen signals?** These comparisons use identical evaluation frames.
* Camera-motion update on natural footage: line alignment rose 0.50→0.88, 0.22→0.82, 0.21→0.80 and 0.24→0.82 across the 4 segments.
  The off-pitch share fell or stayed the same.
* Densify sampling: short-track ratio fell 0.41→0.26 (3x clip, seg2) and 0.47→0.36 (5x clip); fragmentation fell 2.25→2.10.
  One densify attempt was rejected, at 1.70→1.55 (below the 10% bar). It escalated as designed.
* Retry limit: at most 2 automatic attempts in any segment of any case. Both segments that reached 2 escalated with a proposed
  stride for human approval.

**Calibration reprojection on held-out landmarks** (manual labels, `eval/labels/fg_landmarks.json`):
* Frame 0, two goal posts not used for fitting: RMSE **3.07 m** (2.43 m and 3.60 m).
* Frames 150, 300 and 450 after the camera drifted: static calibration RMSE **7.88 m**; motion-updated calibration **0.63 m**
  (16 points). These points re-annotate the calibration landmarks on later frames, so they isolate motion error.

**Camera-motion accuracy** (synthetic pan with known homography): 24/24 probe frames validated. The median of the per-frame max
corner error was **0.46 px**, p95 0.65 px, at 1920×1080.

**Identity consistency** (8 people, 46 labelled boxes, frames 0–90; labels made by the assistant, not by a human):
coverage 0.978, **1 ID switch**, 0.125 extra IDs per person, and no track covering two people. Results were identical for the
initial and corrected runs, because drift in this segment is small (≤54 px) and tracking runs in image space.

**Processing time:** natural 16.9 s clip, agent stage 125 s (0.70 s per analysed frame, including reruns and motion
estimation). The full HTTP run in Docker (linux/arm64, worker capped at 3.5 CPUs), including rendering two annotated videos, took 281 s. Detection is about 0.2 s per
1080p frame (3–4 tiles). EC2 timing is **not measured**.

## 8. Failure cases

1. **Wrong-but-consistent calibration (shifted_points; synthetic).** All four points were shifted 40 px. The homography is
   valid and the line-alignment signal compares frames with the *calibration frame*, so it cannot detect the error.
   Positions are biased by several metres. Mitigations: held-out check points in the calibration UI (reported in metres),
   and future line-to-template alignment.
2. **Texture loss (textureless_pan; synthetic).** Motion still "validated" on residual line and player texture, but the update
   did not improve alignment, so COMPARE rejected it and the agent escalated. The failure was caught at COMPARE rather than
   by the motion-validation gate.
3. **Scene cut (scene_cut; synthetic edit of two real clips).** After the cut, the reference frame cannot be matched, so the agent
   escalates for recalibration. This is correct, but it means multi-shot clips need one calibration per shot.
4. **Container metadata (natural, Latvia CC0 clip).** The header claims 120 fps, so a 16.9 s clip read as 4.2 s and would
   have been rejected. It is fixed by timestamp-derived fps. That clip also shows almost no pitch (13–55% grass), and no
   marked rectangle is fully visible for calibration.
5. **Spectator leakage (natural).** Photographers at the grass edge passed the original foot-on-grass test. The raw-grass-under-foot
   check removes most of them; some edge cases remain and are removed later by the off-pitch projection test.
6. **Far-field extrapolation (natural).** With a near-goal calibration, 38–54% of team points lie more than 30 m from the
   calibrated region, and the far goal projects more than 30 m off. Width and depth values carry this caveat.
7. **Sped-up motion (time_compressed; synthetic).** Inflated apparent speed keeps the jump rate high even after densifying, so
   segments end in review rather than full recovery. This is the right outcome for unreliable tracking.

## Limitations

* **Calibration support.** The four-point homography is exact only inside the calibrated quad. In the demo, the far goal is
  misplaced by more than 30 m, and 38–54% of team points lie more than 30 m from the calibrated rectangle. Width/depth values carry that
  caveat in the UI. Adding more landmarks (least squares) and lens-distortion modelling are the next steps.
* **Line alignment is relative to the calibration frame.** It detects *changes* in geometry (camera motion), not a calibration
  that was wrong from the start. The `shifted_points` case shows this blind spot.
* **Identity:** no re-ID; IDs are anonymous and can switch or re-spawn. We do not claim individual identity.
* **Teams:** colour clustering under stadium light is fragile for dark or grey kits; confidence is a heuristic agreement
  score. Users can set kit colours and override tracks.
* **Partial view:** width/depth describe visible players only (median about 9 of 11 in the demo), not full-team shape.
* **Not implemented:** ball tracking, possession, passing, speed and distance. The UI lists these as unavailable.
* **Spectator filtering** relies on a grass mask. Photographers kneeling on the grass edge can pass the foot-on-grass test; most
  are removed later by the off-pitch projection test.
* **Throughput:** about 0.37 s per analysed 1080p frame on an Apple M3 CPU. EC2 performance is unmeasured because the deployment was not run.
* Thresholds were tuned on one natural clip plus synthetic stress tests. They are not validated across venues.

## Responsible use

* Outputs are approximate and are labelled as measured, heuristic, or unavailable. They should inform, not replace, coaching
  judgement. They are not suitable for player evaluation or recruitment decisions on their own.
* The system does not identify individuals. Do not use it to profile specific players without their knowledge, and follow your
  institution's policies on filming athletes, especially minors.
* Raw clips stay private: the S3 bucket blocks public access and enforces TLS, SSE and 7-day expiry, and the API is token-protected and
  restricted by IP. Upload only footage you have the right to use. The demo uses only CC-licensed footage, with attribution.
