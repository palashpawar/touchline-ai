"""Assemble the narrated demo from recorded frames + narration audio.

    <venv-with-playwright>/bin/python video/assemble.py     ->  video/touchline-ai-demo.mp4
"""
import asyncio
import json
import subprocess
from pathlib import Path

from playwright.async_api import async_playwright

VID = Path(__file__).resolve().parent
ROOT = VID.parent
FR = VID / "frames"
BUILD = VID / "build"
SCENES = ["s1_hook", "s2_upload", "s3_wrong", "s3_right", "s4_run", "s5_agent", "s6_proof", "s7_tactics", "s8_results", "s9_close"]
TITLE_S = 4.5      # title card shown over the opening line
GAP_MAX, GAP_TO = 2.0, 0.8   # compress network waits (upload) in clamp_gaps scenes


def dur(p):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)]))


async def render_cards():
    logo = (ROOT / "assets/logo-3x2.svg").read_text().replace("<svg ", '<svg preserveAspectRatio="xMidYMid slice" width="1920" height="1080" ', 1)
    icon = (ROOT / "assets/logo-icon.svg").read_text()
    end = f"""<html><body style="margin:0;width:1920px;height:1080px;background:radial-gradient(1400px 800px at 50% 20%,#14294b,#0a1120);
      font-family:-apple-system,'Helvetica Neue',Arial,sans-serif;color:#e6edf7;display:flex;flex-direction:column;align-items:center;justify-content:center">
      <div style="width:220px">{icon}</div>
      <div style="font-size:96px;font-weight:600;letter-spacing:-2px;margin-top:18px">Touchline <span style="color:#22c55e;font-weight:800">AI</span></div>
      <div style="font-size:38px;color:#8ea0bd;margin-top:6px">Analysis that knows when it&#8217;s wrong.</div>
      <div style="margin-top:56px;font-size:34px;line-height:1.7;text-align:center">
        <div>&#9654; Live demo &nbsp;<b style="color:#4ade80">palashpawar.github.io/touchline-ai</b></div>
        <div>&lt;/&gt; Code &nbsp;<b style="color:#4ade80">github.com/palashpawar/touchline-ai</b></div></div>
      <div style="position:absolute;bottom:42px;font-size:22px;color:#8ea0bd;text-align:center;line-height:1.6">
        OpenCV 5.0.0 &middot; YOLOX-S (Apache-2.0, OpenCV Zoo) &middot; Runs on AWS EC2 + S3 &middot; Narration: macOS text-to-speech (Daniel)<br>
        Match footage: &#8220;Match de football France-Allemagne&#8221; by Like tears in rain, CC BY-SA 4.0, Wikimedia Commons (annotations added)</div>
      </body></html>"""
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome")
        pg = await b.new_page(viewport={"width": 1920, "height": 1080})
        await pg.set_content(f'<html><body style="margin:0">{logo}</body></html>')
        await pg.screenshot(path=str(BUILD / "title.jpg"), type="jpeg", quality=95)
        await pg.set_content(end)
        await pg.screenshot(path=str(BUILD / "end.jpg"), type="jpeg", quality=95)
        await b.close()


def scene_list(scene):
    """Return [(path, seconds)] for one scene, holding each frame until the next (last until scene end)."""
    if scene == "s9_close":
        return [(BUILD / "end.jpg", dur(VID / "audio/s9_close.aiff") + 2.0)]
    m = json.loads((FR / f"{scene}.json").read_text())
    frames, end = m["frames"], m["end"] - m["t0"]
    items = []
    for i, (t, name) in enumerate(frames):
        nxt = frames[i + 1][0] if i + 1 < len(frames) else end
        d = nxt - t
        if m["clamp_gaps"] and d > GAP_MAX:
            d = GAP_TO
        if d <= 0:
            continue  # duplicate timestamp; the fps filter resamples to 30 fps anyway
        items.append([FR / scene / name, d])
    if scene == "s1_hook":  # title card over the first line, then the app
        shown, kept = 0.0, []
        for path, d in items:
            if shown + d <= TITLE_S:
                shown += d
                continue
            kept.append([path, d - max(0.0, TITLE_S - shown)])
            shown = TITLE_S
        items = [[BUILD / "title.jpg", TITLE_S]] + kept
    # never shorter than the narration
    need = dur(VID / "audio" / f"{scene}.aiff") + 0.6
    total = sum(d for _, d in items)
    if total < need:
        items[-1][1] += need - total
    return items


def encode_video(items, out):
    """Video-only encode of [(image, seconds)] that all share one source resolution."""
    lst = out.with_suffix(".txt")
    lines = []
    for path, d in items:
        lines += [f"file '{path}'", f"duration {d:.4f}"]
    lines.append(f"file '{items[-1][0]}'")  # concat demuxer needs the last file repeated
    lst.write_text("\n".join(lines) + "\n")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-vf", "scale=1920:1080:flags=lanczos,fps=30,format=yuv420p", "-t", f"{sum(d for _, d in items):.3f}",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-an", str(out)], check=True)


def build_scene(scene):
    items = scene_list(scene)
    total = sum(d for _, d in items)
    # Cards (1920x1080) and screencast frames (1536x864) are encoded as separate runs: ffmpeg's concat
    # demuxer drops frames when the source resolution changes mid-list.
    runs = []
    for path, d in items:
        key = Path(path).parent == BUILD
        if runs and runs[-1][0] == key:
            runs[-1][1].append((path, d))
        else:
            runs.append((key, [(path, d)]))
    parts = []
    for i, (_, run) in enumerate(runs):
        part = BUILD / f"{scene}_part{i}.mp4"
        encode_video(run, part)
        parts.append(part)
    vlist = BUILD / f"{scene}_parts.txt"
    vlist.write_text("".join(f"file '{p}'\n" for p in parts))
    out = BUILD / f"{scene}.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(vlist),
                    "-i", str(VID / "audio" / f"{scene}.aiff"),
                    "-filter_complex", f"[1:a]aresample=48000,apad,atrim=0:{total:.3f}[a]",
                    "-map", "0:v", "-map", "[a]", "-t", f"{total:.3f}", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2", str(out)], check=True)
    print(f"{scene}: {total:5.1f}s")
    return out


def main():
    BUILD.mkdir(exist_ok=True)
    asyncio.run(render_cards())
    parts = [build_scene(s) for s in SCENES]
    (BUILD / "all.txt").write_text("".join(f"file '{p}'\n" for p in parts))
    joined = BUILD / "joined.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(BUILD / "all.txt"), "-c", "copy", str(joined)], check=True)
    total = dur(joined)
    final = VID / "touchline-ai-demo.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(joined),
                    "-vf", f"fade=t=in:st=0:d=0.6,fade=t=out:st={total - 0.8:.2f}:d=0.8",
                    "-af", f"afade=t=in:st=0:d=0.3,afade=t=out:st={total - 0.8:.2f}:d=0.8",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-c:a", "aac", "-b:a", "160k",
                    "-movflags", "+faststart", str(final)], check=True)
    print(f"final: {final} {dur(final):.1f}s {final.stat().st_size / 1e6:.1f} MB")


main()
