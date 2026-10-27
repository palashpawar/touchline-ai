"""Record the narrated demo: drive the live app scene by scene and capture frames with Chrome's screencast.

    <venv-with-playwright>/bin/python video/record_demo.py http://<EC2-IP>

Writes video/frames/<scene>/NNNNN.jpg plus video/frames/<scene>.json (frame timestamps, scene length).
Each scene runs its actions, then holds until its narration (video/audio/<scene>.aiff) has finished.
"""
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

from playwright.async_api import async_playwright

BASE = sys.argv[1].rstrip("/")
ROOT = Path(__file__).resolve().parents[1]
VID = ROOT / "video"
CLIP = ROOT / "data/clips/france_germany_2018_phase1_ccbysa.ogv"
W, H = 1536, 864  # CSS viewport; captured at 1.25x = 1920x1080 for larger, crisp UI text
CAL_POINTS = [(520, 836), (640, 811), (1190, 795), (928, 873)]  # frame-0 landmarks (eval/labels/fg_landmarks.json)

CURSOR_JS = """
(() => {
  const add = () => {
    if (document.getElementById('demo-cursor')) return;
    const c = document.createElement('div'); c.id = 'demo-cursor';
    c.style.cssText = 'position:fixed;left:-50px;top:-50px;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;' +
      'background:rgba(255,255,255,.92);border:2px solid #0a1120;box-shadow:0 0 0 4px rgba(34,197,94,.45);z-index:2147483647;pointer-events:none;';
    document.body.appendChild(c);
    document.addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
    document.addEventListener('mousedown', e => {
      const r = document.createElement('div');
      r.style.cssText = `position:fixed;left:${e.clientX}px;top:${e.clientY}px;width:16px;height:16px;margin:-8px 0 0 -8px;` +
        'border-radius:50%;border:3px solid #22c55e;z-index:2147483646;pointer-events:none;transition:all .45s ease-out;opacity:1;';
      document.body.appendChild(r);
      requestAnimationFrame(() => { r.style.width = r.style.height = '56px'; r.style.margin = '-28px 0 0 -28px'; r.style.opacity = '0'; });
      setTimeout(() => r.remove(), 600);
    }, true);
  };
  if (document.body) add(); else document.addEventListener('DOMContentLoaded', add);
})();
"""


def narration_len(scene: str) -> float:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                   str(VID / "audio" / f"{scene}.aiff")])
    return float(out)


class Recorder:
    def __init__(self, page, cdp):
        self.page, self.cdp = page, cdp
        self.scene, self.frames, self.dir = None, [], None
        cdp.on("Page.screencastFrame", self._on_frame)

    def _on_frame(self, ev):
        asyncio.ensure_future(self.cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]}))
        if self.scene is None:
            return
        import base64
        p = self.dir / f"{len(self.frames):05d}.jpg"
        p.write_bytes(base64.b64decode(ev["data"]))
        self.frames.append((time.monotonic(), p.name))

    async def start(self, scene):
        self.scene, self.frames = scene, []
        self.dir = VID / "frames" / scene
        self.dir.mkdir(parents=True, exist_ok=True)
        for f in self.dir.glob("*.jpg"):
            f.unlink()
        self.t0 = time.monotonic()
        # a 1-px nudge forces a fresh frame at scene start even if nothing moves
        await self.page.mouse.move(*self.mouse, steps=1)

    async def hold_until_narrated(self, pad=0.9):
        remaining = narration_len(self.scene) + pad - (time.monotonic() - self.t0)
        if remaining > 0:
            await asyncio.sleep(remaining)

    def stop(self, clamp_gaps=False):
        end = time.monotonic()
        meta = {"scene": self.scene, "t0": self.t0, "end": end, "clamp_gaps": clamp_gaps,
                "frames": [[t - self.t0, n] for t, n in self.frames]}
        (VID / "frames" / f"{self.scene}.json").write_text(json.dumps(meta))
        print(f"{self.scene}: {len(self.frames)} frames, {end - self.t0:.1f}s")
        self.scene = None


async def rec_pause(rec):
    rec.scene = None  # frames between scenes are ignored


async def move_click(page, rec, x, y, steps=18, pause=0.25):
    await page.mouse.move(x, y, steps=steps)
    rec.mouse = (x, y)
    await asyncio.sleep(pause)
    await page.mouse.click(x, y)


async def click_el(page, rec, locator, pause=0.25):
    await locator.scroll_into_view_if_needed()
    b = await locator.bounding_box()
    await move_click(page, rec, b["x"] + b["width"] / 2, b["y"] + b["height"] / 2, pause=pause)


async def hover_el(page, rec, locator, dwell=1.0):
    b = await locator.bounding_box()
    x, y = b["x"] + b["width"] / 2, b["y"] + b["height"] / 2
    await page.mouse.move(x, y, steps=20)
    rec.mouse = (x, y)
    await asyncio.sleep(dwell)


async def smooth_scroll(page, y, ms=900):
    await page.evaluate("""([y, ms]) => new Promise(r => { const s = window.scrollY, t0 = performance.now();
        const f = now => { const k = Math.min(1, (now - t0) / ms), e = k < .5 ? 2*k*k : 1 - Math.pow(-2*k + 2, 2) / 2;
          window.scrollTo(0, s + (y - s) * e); k < 1 ? requestAnimationFrame(f) : r(); }; requestAnimationFrame(f); })""", [y, ms])


async def scroll_card(page, heading, offset=90, ms=1000):
    y = await page.evaluate("""([h, off]) => { const el = [...document.querySelectorAll('.card-head h3')].find(e => e.textContent.startsWith(h));
        return el.closest('.card').getBoundingClientRect().top + window.scrollY - off; }""", [heading, offset])
    await smooth_scroll(page, y, ms)


async def show_video(page, label, t=7.2):
    """Switch the results video (Corrected / Initial run) and wait until the frame at t is decoded."""
    await page.locator(".seg-ctl button", has_text=label).click()
    await page.wait_for_function("document.querySelector('video') && document.querySelector('video').readyState >= 1", timeout=120_000)
    await page.evaluate("""t => new Promise(r => { const v = document.querySelector('video'); v.muted = true; v.pause();
        const done = () => { if (v.readyState >= 3) setTimeout(r, 200); else v.addEventListener('canplay', () => setTimeout(r, 200), {once: true}); };
        v.addEventListener('seeked', done, {once: true}); v.currentTime = t; })""", t)


async def cal_click(page, rec, idx):
    svg = page.locator(".frame-wrap svg")
    b = await svg.bounding_box()
    px, py = CAL_POINTS[idx]
    await move_click(page, rec, b["x"] + px / 1920 * b["width"], b["y"] + py / 1080 * b["height"], steps=14, pause=0.3)
    await asyncio.sleep(0.35)


async def main():
    token = subprocess.check_output(["aws", "ssm", "get-parameter", "--region", "us-east-2", "--name", "/touchline/api_token",
                                     "--with-decryption", "--query", "Parameter.Value", "--output", "text"], text=True).strip()
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome")
        ctx = await b.new_context(viewport={"width": W, "height": H}, device_scale_factor=1.25)
        await ctx.add_init_script(CURSOR_JS)
        page = await ctx.new_page()
        cdp = await ctx.new_cdp_session(page)
        rec = Recorder(page, cdp)
        rec.mouse = (W / 2, H / 2)

        # --- log in (not recorded) ---
        await page.goto(BASE, wait_until="networkidle")
        if await page.locator("input[type=password]").count():
            await page.fill("input[type=password]", token)
            await page.click("button.primary")
        await page.wait_for_selector(".dropzone")
        await page.add_style_tag(content=".tl-item{animation:none!important}")
        await cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92, "maxWidth": 1920, "maxHeight": 1080, "everyNthFrame": 1})
        await asyncio.sleep(0.5)

        # s1: hook on the upload page
        await rec.start("s1_hook")
        await hover_el(page, rec, page.locator(".hero h1"), 2.0)
        await hover_el(page, rec, page.locator(".dropzone"), 1.5)
        await rec.hold_until_narrated(pad=0.4)
        rec.stop()

        # s2: upload (long network waits are compressed when assembling)
        await rec.start("s2_upload")
        await hover_el(page, rec, page.locator(".dropzone"), 0.8)
        async with page.expect_file_chooser() as fc:
            await page.mouse.click(*rec.mouse)
        await (await fc.value).set_files(str(CLIP))
        await page.wait_for_selector(".frame-wrap img", timeout=600_000)
        await page.wait_for_function("(() => { const i = document.querySelector('.frame-wrap img'); return i && i.complete && i.naturalWidth > 0; })()", timeout=120_000)
        await asyncio.sleep(1.2)
        await rec.hold_until_narrated()
        rec.stop(clamp_gaps=True)

        # s3a: calibration in the wrong order -> rejected
        await rec.start("s3_wrong")
        await click_el(page, rec, page.locator(".calib-side .seg-ctl button", has_text="4 landmarks"))
        await page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(0.6)
        for i in (0, 2, 1, 3):
            await cal_click(page, rec, i)
        await click_el(page, rec, page.locator("button", has_text="Validate calibration"))
        await page.wait_for_selector(".status-pill.bad", timeout=30_000)
        await hover_el(page, rec, page.locator(".calib-report .alert.error").first, 1.5)
        await rec.hold_until_narrated()
        rec.stop()

        # s3b: right order -> accepted, template projected
        await rec.start("s3_right")
        await click_el(page, rec, page.locator("button", has_text="Reset points"))
        await page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(0.5)
        for i in range(4):
            await cal_click(page, rec, i)
        await click_el(page, rec, page.locator("button", has_text="Validate calibration"))
        await page.wait_for_selector(".status-pill.ok", timeout=30_000)
        await page.evaluate("window.scrollTo(0, 0)")
        await hover_el(page, rec, page.locator(".frame-wrap"), 2.0)
        await rec.hold_until_narrated()
        rec.stop()

        # s4: run on AWS, show the first agent steps arriving
        await rec.start("s4_run")
        await click_el(page, rec, page.locator("button.stp", has_text="Analyze"))
        await asyncio.sleep(1.0)
        await click_el(page, rec, page.locator("button.primary.big"))
        await asyncio.sleep(9.0)
        await rec.hold_until_narrated()
        rec.stop()

        # s5: cut to the finished run; walk through the agent's seg2 decision
        await page.locator("button.stp", has_text="Upload").click()
        await page.locator("table.jobs tr", has_text="complete").last.click()  # the original run the narration describes
        await page.wait_for_selector(".timeline-card .tl-item")
        await page.wait_for_selector("video")
        await page.evaluate("""() => new Promise(r => { const v = document.querySelector('video'); v.muted = true; v.pause();
            v.addEventListener('seeked', () => setTimeout(r, 300), {once: true}); v.currentTime = 7.2; })""")
        await asyncio.sleep(0.8)
        await rec.start("s5_agent")
        tl = page.locator(".timeline-card")
        await asyncio.sleep(2.0)
        await click_el(page, rec, tl.locator(".seg-ctl button", has_text="seg2"))
        await asyncio.sleep(2.5)
        head = tl.locator(".tl-head", has_text="camera_motion_update").first
        await click_el(page, rec, head)
        await page.evaluate("""() => { const c = document.querySelector('.timeline-card'); const it = c.querySelector('.tl-item.open');
            c.scrollTo({top: it.offsetTop - c.offsetTop - 60, behavior: 'smooth'}); }""")
        await asyncio.sleep(7.0)
        await click_el(page, rec, head)  # collapse
        upd = tl.locator(".tl-head", has_text="update_calibration").first
        await upd.scroll_into_view_if_needed()
        await asyncio.sleep(0.6)
        await click_el(page, rec, upd)
        await asyncio.sleep(6.0)
        await click_el(page, rec, upd)
        cmp_ = tl.locator(".tl-head", has_text="compare_runs").first
        await cmp_.scroll_into_view_if_needed()
        await asyncio.sleep(0.6)
        await click_el(page, rec, cmp_)
        await page.evaluate("""() => { const c = document.querySelector('.timeline-card'); const it = c.querySelector('.tl-item.open');
            c.scrollTo({top: it.offsetTop - c.offsetTop - 60, behavior: 'smooth'}); }""")
        await asyncio.sleep(5.0)
        await rec.hold_until_narrated()
        rec.stop()

        # s6: same-frame before/after, then flip the video between runs.
        # Pre-load both runs' videos off camera so the switch on camera is instant (no loading spinner).
        await rec_pause(rec)
        await show_video(page, "Initial run")
        await show_video(page, "Corrected")
        await rec.start("s6_proof")
        await scroll_card(page, "Before / after", 90, 1200)
        await smooth_scroll(page, await page.evaluate("window.scrollY") + 430, 700)
        await hover_el(page, rec, page.locator(".compare-list figure").nth(1), 3.0)
        await smooth_scroll(page, 0, 1000)
        btn = page.locator(".seg-ctl button", has_text="Initial run")
        await hover_el(page, rec, btn, 0.3)
        await page.mouse.down(); await page.mouse.up()
        await show_video(page, "Initial run")
        await asyncio.sleep(2.2)
        btn = page.locator(".seg-ctl button", has_text="Corrected")
        await hover_el(page, rec, btn, 0.3)
        await page.mouse.down(); await page.mouse.up()
        await show_video(page, "Corrected")
        await asyncio.sleep(1.5)
        await rec.hold_until_narrated()
        rec.stop()

        # s7: heatmap + metrics with caveats + not measured
        await rec.start("s7_tactics")
        sel = page.locator(".card-head select")
        await hover_el(page, rec, sel, 0.4)
        await sel.select_option("all")
        await hover_el(page, rec, page.locator(".pitch-svg").first, 3.0)
        await scroll_card(page, "Tactical measurements", 90, 1100)
        await hover_el(page, rec, page.locator(".metric-card").nth(0), 2.5)
        await hover_el(page, rec, page.locator(".metric-card").nth(2), 3.0)
        await rec.hold_until_narrated()
        rec.stop()

        # s8: evaluation table + S3 evidence on GitHub
        await rec.start("s8_results")
        await page.goto("https://github.com/palashpawar/touchline-ai/blob/master/eval/RESULTS.md", wait_until="domcontentloaded")
        await page.wait_for_selector("article table", timeout=30_000)
        await asyncio.sleep(1.0)
        y = await page.evaluate("document.querySelector('article table').getBoundingClientRect().top + window.scrollY - 140")
        await smooth_scroll(page, y, 1200)
        await hover_el(page, rec, page.locator("article table").first, 6.0)
        await page.goto("https://github.com/palashpawar/touchline-ai/blob/master/docs/demo/aws/s3_listing.txt", wait_until="domcontentloaded")
        await page.wait_for_timeout(2500)
        await page.mouse.move(W / 2, H / 2, steps=10)
        await rec.hold_until_narrated()
        rec.stop()

        await cdp.send("Page.stopScreencast")
        await b.close()


asyncio.run(main())
