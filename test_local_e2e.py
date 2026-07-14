"""
Local end-to-end verification: OLD pipeline vs NEW pipeline on the same clip.

OLD = whisper tiny int8 CPU, per-segment Google Translate, single-line drawtext
NEW = whisper large-v3-turbo CUDA + VAD + word timestamps, Claude chunked
      translation, sentence-segmented 2-line cues

Usage:  python test_local_e2e.py <source_video> <start_sec> <dur_sec> <target_lang>
Writes everything to jobs/local_e2e/.
"""
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import processor  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else r"T:\watch_videos\Logic_Pro_A_Complete_Tutorial_For_The_Overwhelmed_Beginner.mp4"
START = sys.argv[2] if len(sys.argv) > 2 else "0"
DUR = sys.argv[3] if len(sys.argv) > 3 else "60"
TARGET = sys.argv[4] if len(sys.argv) > 4 else "es"

JOB_ID = "local_e2e"
JOB_DIR = os.path.join(processor.JOBS_DIR, JOB_ID)
os.makedirs(JOB_DIR, exist_ok=True)
CLIP = os.path.join(JOB_DIR, "input.mp4")
FFMPEG = processor._get_ffmpeg()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── 1. cut the clip ───────────────────────────────────────────────────────────
log(f"Cutting {DUR}s clip from {SRC} @ {START}s")
subprocess.run([FFMPEG, "-y", "-ss", START, "-t", DUR, "-i", SRC,
                "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", CLIP],
               check=True, capture_output=True)

# ── 2. OLD pipeline ───────────────────────────────────────────────────────────
log("OLD pipeline: whisper tiny (cpu int8) + GoogleTranslator ...")
t0 = time.time()
from faster_whisper import WhisperModel  # noqa: E402
from deep_translator import GoogleTranslator  # noqa: E402

old_model = WhisperModel("tiny", device="cpu", compute_type="int8")
old_segments = list(old_model.transcribe(CLIP, language="en")[0])
del old_model
old_lines = []
for seg in old_segments:
    text = seg.text.strip()
    translated = GoogleTranslator(source="auto", target=TARGET).translate(text)
    old_lines.append((seg.start, seg.end, text, translated))
old_wall = time.time() - t0

old_srt = os.path.join(JOB_DIR, "old_subtitles.srt")
with open(old_srt, "w", encoding="utf-8") as f:
    for i, (s, e, _src, tr) in enumerate(old_lines, 1):
        f.write(f"{i}\n{processor.fmt_time(s)} --> {processor.fmt_time(e)}\n{tr}\n\n")

# old-style single-line drawtext burn (replicates the retired pipeline)
def _old_esc(t):
    return (t.replace("\\", "\\\\").replace("'", "’")
             .replace(":", "\\:").replace("%", "\\%"))

old_filters = []
font_part = ""
for cand in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", r"C:\Windows\Fonts\arial.ttf"]:
    if os.path.exists(cand):
        font_part = ":fontfile='" + cand.replace("\\", "/").replace(":", "\\:") + "'"
        break
for s, e, _src, tr in old_lines:
    txt = _old_esc(tr)
    old_filters.append(
        f"drawtext=text='{txt}'{font_part}:x=(w-text_w)/2:y=h-80"
        f":fontsize=24:fontcolor=white:borderw=2:bordercolor=black"
        f":enable='between(t,{s},{e})'")
old_script = os.path.join(JOB_DIR, "old_filter.txt")
with open(old_script, "w", encoding="utf-8") as f:
    f.write(",\n".join(old_filters))
old_out = os.path.join(JOB_DIR, "old_output.mp4")
subprocess.run([FFMPEG, "-y", "-i", CLIP, "-filter_script:v", old_script,
                "-c:a", "copy", old_out], check=True, capture_output=True)
log(f"OLD done in {old_wall:.1f}s transcribe+translate -> {old_out}")

# ── 3. NEW pipeline (the real production path) ────────────────────────────────
log("NEW pipeline: process_job() ...")
t0 = time.time()
processor.jobs[JOB_ID] = {"status": "queued", "progress": 0, "message": "", "output_path": None}
processor.process_job(JOB_ID, "auto", TARGET)
new_wall = time.time() - t0
job = processor.jobs[JOB_ID]
log(f"NEW done in {new_wall:.1f}s  status={job['status']}  msg={job['message']}")
if job["status"] != "done":
    sys.exit(1)

# ── 4. frame extraction for visual QC ─────────────────────────────────────────
srt = open(os.path.join(JOB_DIR, "subtitles.srt"), encoding="utf-8").read()
stamps = re.findall(r"(\d+):(\d+):(\d+),(\d+) --> (\d+):(\d+):(\d+),(\d+)", srt)
picks = [stamps[i] for i in (0, len(stamps) // 2, len(stamps) - 1)] if stamps else []
for n, st in enumerate(picks):
    h, m, s, ms, h2, m2, s2, ms2 = map(int, st)
    a = h * 3600 + m * 60 + s + ms / 1000
    b = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
    mid = (a + b) / 2
    for tag, vid in (("new", os.path.join(JOB_DIR, "output.mp4")), ("old", old_out)):
        png = os.path.join(JOB_DIR, f"qc_{tag}_{n}.png")
        subprocess.run([FFMPEG, "-y", "-ss", f"{mid:.2f}", "-i", vid,
                        "-frames:v", "1", png], check=True, capture_output=True)
log("QC frames written: " + JOB_DIR)

# ── 5. side-by-side line comparison ───────────────────────────────────────────
print("\n===== OLD (tiny + Google) =====")
for s, e, src, tr in old_lines[:8]:
    print(f"  [{s:6.2f}-{e:6.2f}] SRC: {src}")
    print(f"                  OLD: {tr}")
print("\n===== NEW (turbo + Claude) — subtitles.srt =====")
print("\n".join(srt.splitlines()[:40]))
print(f"\nWALL: old={old_wall:.1f}s  new={new_wall:.1f}s")
print(f"Claude usage: {__import__('translate_llm').usage_totals}")
