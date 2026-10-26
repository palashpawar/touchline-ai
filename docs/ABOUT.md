## Inspiration

College soccer teams film from the stands on a phone. I wanted to turn that footage into tactics coaches use, like team
shape, width and heatmaps. Then I noticed something uncomfortable. In my test clip, the camera drifts **135 pixels** from
where I calibrated it. Nothing breaks, but every player position quietly slides several metres. **Numbers that look right
but aren't are worse than no numbers.** So Touchline AI became analysis that _knows when it's wrong_.

## What it does

You upload a 10–30 s clip and mark four points of a known pitch area. An agent then works through the clip in ~4 s
segments:

1. **Perceive:** detect and track players, project them onto a top-down pitch, and measure its own quality.
2. **Decide:** choose a fix based on _which_ check failed (camera moved → update the calibration; tracking broke up →
   sample frames more densely).
3. **Act:** rerun that segment and keep the fix only if it measurably helps on the **same frames**.
4. **Escalate:** after two failed attempts, keep the old result, flag the segment, and ask the coach.

The agent is rule-based, with no LLM in the loop, so every decision is reproducible and logged.

## How I built it

**OpenCV 5** does the vision work: decoding, grass segmentation, line detection, calibration, camera motion, Kalman
tracking, and running a YOLOX-S player detector through `cv2.dnn` on tiles (players are only 30–60 px tall).

Calibration solves a homography from four image points to known pitch coordinates in metres:

> **(x, y, 1) ∝ H · (u, v, 1)**  (pixel → pitch metres)

When the camera moves, I estimate each frame's motion (ORB features or optical flow, fitted with MAGSAC) and update the
calibration frame by frame:

> **Hₜ = H(ref) · Mₜ**

To check that the geometry still fits, I warp each frame's painted lines into the calibration frame and measure the share
that lands within 4 px of a reference line:

> **alignment = (line pixels within 4 px of a reference line) ÷ (all line pixels)**, where 1.0 means a perfect fit

A rerun is kept only if a target signal improves by at least 10% and nothing else gets worse.

**Stack:** FastAPI, SQLite and a background worker; React/TypeScript; Docker. On **AWS**, an EC2 worker processes clips
from S3, with encrypted storage, a narrowly scoped IAM role and automatic cleanup.

## Results

* On a real handheld clip, it fixed camera drift in all 4 segments. Line alignment went from 0.2–0.5 to 0.8–0.9.
* After the camera moved, landmark position error dropped from **7.9 m to 0.6 m**.
* It chose the right action in **8 of 9** stress tests.
* It runs end to end on EC2 + S3.

## Challenges

* **My quality check fooled me.** A static clip got flagged because photographers at the pitch edge slipped past the
  spectator filter. I fixed the filter itself (requiring grass under each person's feet) rather than loosening the
  threshold.
* **Checks that fought each other.** On sped-up footage, a valid camera fix was vetoed because players _looked_ too fast. I
  judged geometry fixes on geometry alone and let the next strategy handle tracking.
* **Files that lie.** One video's header claimed 120 fps, so a 17 s clip looked 4 s long. I now read frame timing from the
  decoded frames.
* **Real-world AWS.** I hit a region lock, free-tier-only instances, and an IP that kept changing. I scripted around each one.

## What I learned

* Saying _"I'm not sure about this part"_ is a feature, not a failure.
* Compare fixes on identical frames, or "better" just means "more data".
* Be explicit about what you **don't** measure. Possession, passing and speed are listed as not measured.
* Test your tests: several early "failures" were bugs in my evaluation code.

## What's next

Catching calibrations that are wrong from the start, using more pitch landmarks, adding player re-identification, and
having humans verify my small evaluation label set (currently AI-assisted).
