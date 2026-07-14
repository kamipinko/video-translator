"""
Job processor: GPU whisper transcription -> Claude translation -> drawtext burn.

2026-07 overhaul:
  - ASR: faster-whisper large-v3-turbo on CUDA float16 with VAD and word
    timestamps (loader pattern from claude-vision/audio.py, incl. the pip
    nvidia-* DLL registration). ~15x realtime on the 5090.
    STRICT: if WHISPER_DEVICE=cuda (the default) and CUDA can't init, the job
    FAILS LOUDLY — no silent CPU fallback. Railway/CPU deploys must set
    WHISPER_DEVICE=cpu and WHISPER_MODEL=tiny explicitly (see DEPLOY_NOTES.md).
  - Translation: Claude (translate_llm.py) in context-aware chunks — replaces
    the old per-line Google Translate pass. Whisper always runs
    task="transcribe" (native language); Claude handles ALL target languages,
    including English.
  - Subtitles: sentence-boundary segmentation from word timestamps, 1-6s cues,
    2x42-char balanced lines, no orphan words (subtitles.py).
  - Burn: one drawtext per rendered line (no newline-escaping headaches),
    filter passed via -filter_script:v file (no command-line length limit).
"""

import gc
import glob
import os
import subprocess
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(BASE_DIR, "jobs")

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")       # CUDA model
WHISPER_MODEL_CPU = os.environ.get("WHISPER_MODEL_CPU", "tiny")         # CPU model
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "auto")  # 'auto' | 'cuda' | 'cpu'
FONTS_DIR = os.path.join(BASE_DIR, "fonts")

jobs = {}

_model_cache = {"model": None, "key": None}

# populated by _load_whisper(); surfaced in /health and job messages
engine_info = {"device": None, "model": None, "note": None}


def fmt_time(seconds):
    ms = int((seconds % 1) * 1000)
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _get_ffmpeg():
    import shutil
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _get_video_size(ffmpeg_exe, path):
    """(width, height) of the DISPLAYED frame (rotation-aware). Falls back to
    parsing `ffmpeg -i` stderr if ffprobe isn't around (imageio wheel)."""
    import json as _json
    import re as _re
    import shutil as _shutil
    ffprobe = _shutil.which("ffprobe")
    if ffprobe:
        try:
            r = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0",
                 "-show_streams", "-of", "json", path],
                capture_output=True, text=True, errors="replace")
            st = _json.loads(r.stdout)["streams"][0]
            w, h = int(st["width"]), int(st["height"])
            rot = 0
            for sd in st.get("side_data_list", []) or []:
                if "rotation" in sd:
                    rot = int(sd["rotation"])
            rot = rot or int(st.get("tags", {}).get("rotate", 0) or 0)
            if abs(rot) % 180 == 90:
                w, h = h, w
            return w, h
        except Exception as e:
            print(f"[processor] ffprobe size failed ({e}); ffmpeg fallback", flush=True)
    r = subprocess.run([ffmpeg_exe, "-i", path], capture_output=True,
                       text=True, errors="replace")
    m = _re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", r.stderr)
    if not m:
        print("[processor] WARNING: could not probe video size, assuming 1920x1080",
              flush=True)
        return 1920, 1080
    return int(m.group(1)), int(m.group(2))


# ── whisper loading (audio.py v2.0 pattern) ───────────────────────────────────

def _register_cuda_dlls():
    """Put pip nvidia-*/bin dirs on PATH so CTranslate2 finds cuBLAS/cuDNN."""
    bins = []
    try:
        import nvidia  # noqa: F401
        nvidia_root = os.path.dirname(nvidia.__file__)
    except Exception:
        nvidia_root = os.path.join(os.path.dirname(os.__file__), "site-packages", "nvidia")
    for b in glob.glob(os.path.join(nvidia_root, "*", "bin")):
        bins.append(b)
        try:
            os.add_dll_directory(b)
        except Exception:
            pass
    if bins:
        os.environ["PATH"] = os.pathsep.join(bins) + os.pathsep + os.environ.get("PATH", "")
    return bins


def _load_whisper():
    """
    Load whisper per WHISPER_DEVICE. Fallback chain is EXPLICIT and LOUD:
      auto  -> try CUDA float16 (WHISPER_MODEL); on any init failure fall back
               to CPU int8 (WHISPER_MODEL_CPU) and SAY SO (log + engine_info
               note surfaced in the job message / /health). Never silent.
      cuda  -> strict: CUDA or a hard error (no fallback at all).
      cpu   -> CPU int8 directly (Railway / no-GPU deploys).
    """
    from faster_whisper import WhisperModel

    if _model_cache["model"] is not None and _model_cache["key"] == WHISPER_DEVICE:
        return _model_cache["model"]

    if WHISPER_DEVICE in ("auto", "cuda"):
        _register_cuda_dlls()
        try:
            model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
            engine_info.update(device="cuda", model=WHISPER_MODEL, note=None)
            # cache on GPU: model stays resident between jobs for speed
            _model_cache.update(model=model, key=WHISPER_DEVICE)
            return model
        except Exception as e:
            if WHISPER_DEVICE == "cuda":
                # strict mode — surface the failure instead of degrading quality
                raise RuntimeError(
                    f"CUDA whisper init failed ({e}). WHISPER_DEVICE=cuda refuses "
                    "CPU fallback — use WHISPER_DEVICE=auto or cpu."
                ) from e
            note = (f"GPU unavailable ({type(e).__name__}) — transcribed on CPU "
                    f"with whisper '{WHISPER_MODEL_CPU}' (slower, less accurate)")
            print(f"[processor] LOUD FALLBACK: {note}", flush=True)
            engine_info.update(device="cpu", model=WHISPER_MODEL_CPU, note=note)
            return WhisperModel(WHISPER_MODEL_CPU, device="cpu", compute_type="int8")

    # explicit CPU (Railway): int8, not cached (512MB RAM budget)
    engine_info.update(device="cpu", model=WHISPER_MODEL_CPU,
                       note=f"CPU mode (whisper {WHISPER_MODEL_CPU} int8 + VAD)")
    return WhisperModel(WHISPER_MODEL_CPU, device="cpu", compute_type="int8")


# ── libass burn (manga caption style, bundled font) ───────────────────────────

def burn_subtitles(ffmpeg_exe, input_path, cues, output_path):
    """
    Burn cues with libass (.ass, Manga style + pop scale-in animation).
    The font is BUNDLED in fonts/Manga-Regular.ttf and passed via fontsdir,
    so no system-font / fontconfig dependency locally or on Railway.
    ffmpeg runs with cwd=BASE_DIR and relative paths — Windows drive colons
    (C\\:) never reach the filter string.
    """
    if not cues:
        cmd = [ffmpeg_exe, "-y", "-i", input_path, "-c", "copy", output_path]
        result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if result.returncode != 0:
            raise RuntimeError("ffmpeg copy failed: " + result.stderr[-800:])
        return

    from subtitles import cues_to_ass
    video_w, video_h = _get_video_size(ffmpeg_exe, input_path)
    ass_path = os.path.join(os.path.dirname(output_path), "subtitles.ass")
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(cues_to_ass(cues, video_w, video_h))

    rel = lambda p: os.path.relpath(p, BASE_DIR).replace("\\", "/")  # noqa: E731
    vf = f"subtitles={rel(ass_path)}:fontsdir=fonts"
    cmd = [ffmpeg_exe, "-y", "-i", rel(input_path),
           "-vf", vf, "-c:a", "copy", rel(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            errors="replace", cwd=BASE_DIR)
    if result.returncode != 0:
        raise RuntimeError("ffmpeg burn failed: " + result.stderr[-800:])


# ── pipeline ──────────────────────────────────────────────────────────────────

# v2.1: multi-speaker / crosstalk hardening.
#   - VAD threshold lowered 0.5 -> 0.35 (quieter overlapped speech was
#     falling below the speech-probability gate and vanishing)
#   - condition_on_previous_text=False (after a crosstalk section the decoder
#     could context-lock and skip/hallucinate whole 30s windows)
#   - no_speech_threshold raised 0.6 -> 0.8 (crosstalk pushes no_speech prob
#     up; default was discarding real speech segments)
#   - GAP-RESCUE PASS: after the first pass, any VAD speech region whose time
#     is <40% covered by cues gets re-transcribed in isolation (fresh decoder
#     context on just that slice) and the rescued cues merged in.
#   - coverage %% (speech time covered by cues) is measured and surfaced.
VAD_PARAMS = {"threshold": 0.35, "min_speech_duration_ms": 100,
              "min_silence_duration_ms": 700, "speech_pad_ms": 400}
DECODE_PARAMS = {"condition_on_previous_text": False, "no_speech_threshold": 0.8}
RESCUE_MIN_COVER = 0.40   # regions covered less than this get a second pass
RESCUE_MIN_DUR = 0.6      # ignore blips shorter than this
RESCUE_MAX_REGIONS = 12   # safety valve per job

# stats from the last transcribe() run (surfaced in job message + /health)
last_coverage = {}


def _speech_regions(audio):
    """Ground-truth VAD speech regions [(start,end), ...] in seconds."""
    from faster_whisper.vad import get_speech_timestamps, VadOptions
    opts = VadOptions(threshold=VAD_PARAMS["threshold"],
                      min_speech_duration_ms=VAD_PARAMS["min_speech_duration_ms"],
                      min_silence_duration_ms=VAD_PARAMS["min_silence_duration_ms"],
                      speech_pad_ms=VAD_PARAMS["speech_pad_ms"])
    return [(r["start"] / 16000.0, r["end"] / 16000.0)
            for r in get_speech_timestamps(audio, opts)]


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def _region_cover(region, cues):
    s, e = region
    if e - s <= 0:
        return 1.0
    return sum(_overlap(s, e, c["start"], c["end"]) for c in cues) / (e - s)


def _coverage_stats(regions, cues):
    total = sum(e - s for s, e in regions)
    covered = sum(min(e - s, sum(_overlap(s, e, c["start"], c["end"]) for c in cues))
                  for s, e in regions)
    # plain floats — VAD hands back numpy scalars, which break JSON responses
    return {"speech_s": round(float(total), 2), "covered_s": round(float(covered), 2),
            "coverage_pct": round(100.0 * float(covered) / float(total), 1) if total else 100.0}


def transcribe(input_path, source_lang="auto"):
    """Run whisper -> (cues, detected_language). Cues are native-language."""
    from subtitles import build_cues

    model = _load_whisper()
    kwargs = {"vad_filter": True, "vad_parameters": dict(VAD_PARAMS),
              "word_timestamps": True, **DECODE_PARAMS}
    if source_lang and source_lang != "auto":
        kwargs["language"] = source_lang
    segments_gen, info = model.transcribe(input_path, **kwargs)
    segments = list(segments_gen)
    detected = getattr(info, "language", None) or source_lang
    cues = build_cues(segments)

    # ── coverage check + gap rescue ───────────────────────────────────────────
    rescued_n = 0
    try:
        from faster_whisper.audio import decode_audio
        audio = decode_audio(input_path, sampling_rate=16000)
        regions = _speech_regions(audio)
        gaps = [r for r in regions
                if (r[1] - r[0]) >= RESCUE_MIN_DUR
                and _region_cover(r, cues) < RESCUE_MIN_COVER]
        for s, e in gaps[:RESCUE_MAX_REGIONS]:
            pad = 0.15
            a0, a1 = max(0.0, s - pad), e + pad
            chunk = audio[int(a0 * 16000): int(a1 * 16000)]
            if len(chunk) < 16000 // 4:
                continue
            rkw = {"word_timestamps": True, **DECODE_PARAMS}
            if detected and detected != "auto":
                rkw["language"] = detected
            rseg = list(model.transcribe(chunk, **rkw)[0])
            rcues = build_cues(rseg)
            for rc in rcues:
                rc["start"] += a0
                rc["end"] += a0
                # keep only rescues that genuinely add coverage
                dur = rc["end"] - rc["start"]
                overlap_existing = sum(_overlap(rc["start"], rc["end"],
                                                c["start"], c["end"]) for c in cues)
                if dur > 0 and overlap_existing / dur < 0.5:
                    cues.append(rc)
                    rescued_n += 1
        if rescued_n:
            cues.sort(key=lambda c: c["start"])
            print(f"[processor] gap-rescue: recovered {rescued_n} cue(s) in "
                  f"{len(gaps)} low-coverage region(s)", flush=True)
        stats = _coverage_stats(regions, cues)
    except Exception as e:  # coverage layer must never kill a job
        print(f"[processor] coverage/rescue pass failed: {e}", flush=True)
        stats = {"error": str(e)}
    stats["rescued_cues"] = rescued_n
    last_coverage.clear()
    last_coverage.update(stats)

    if engine_info.get("device") != "cuda":
        # CPU path (Railway 512MB): free the model between jobs
        del model
        gc.collect()

    return cues, detected


def process_job(job_id, source_lang, target_lang):
    try:
        job = jobs[job_id]
        job["status"] = "processing"
        job["progress"] = 5
        job["message"] = f"Loading whisper {WHISPER_MODEL} ({WHISPER_DEVICE})..."

        ffmpeg_exe = _get_ffmpeg()
        job_dir = os.path.join(JOBS_DIR, job_id)
        input_path = None
        for f in os.listdir(job_dir):
            if f.startswith("input"):
                input_path = os.path.join(job_dir, f)
                break
        if input_path is None:
            raise FileNotFoundError("No input file found in job directory")

        job["progress"] = 15
        job["message"] = "Transcribing audio (whisper + VAD)..."
        cues, detected = transcribe(input_path, source_lang)
        job["detected_language"] = detected
        job["engine"] = dict(engine_info)
        if engine_info.get("note"):
            # LOUD: CPU fallback / CPU mode is surfaced to the client
            job["message"] = engine_info["note"]

        # Translate with Claude unless target == source language
        if cues and target_lang and target_lang != detected:
            import translate_llm
            job["progress"] = 50
            job["message"] = f"Translating {len(cues)} lines with Claude..."

            def _cb(done, total):
                job["progress"] = 50 + int(20 * done / total)
                job["message"] = f"Translating with Claude ({done}/{total} chunks)..."

            texts = translate_llm.translate_lines(
                [c["text"] for c in cues], target_lang,
                source_lang=detected, progress_cb=_cb,
            )
            # HARD COUNT INVARIANT: one translation per cue, always.
            if len(texts) != len(cues):
                raise RuntimeError(
                    f"translation count invariant broken: {len(cues)} cues -> "
                    f"{len(texts)} translations")
            for cue, t in zip(cues, texts):
                cue["text"] = t
            job["translate_fallbacks"] = translate_llm.last_report.get("fallbacks", [])

        # SRT (with proper 2-line breaks)
        from subtitles import cues_to_srt
        srt_path = os.path.join(job_dir, "subtitles.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(cues_to_srt(cues, fmt_time))

        job["progress"] = 75
        job["message"] = "Burning subtitles..."
        output_path = os.path.join(job_dir, "output.mp4")
        burn_subtitles(ffmpeg_exe, input_path, cues, output_path)

        job["status"] = "done"
        job["progress"] = 100
        job["output_path"] = output_path
        job["coverage"] = dict(last_coverage)
        note = f" [{engine_info['note']}]" if engine_info.get("note") else ""
        cov = last_coverage.get("coverage_pct")
        cov_note = f", speech coverage {cov}%" if cov is not None else ""
        fb = job.get("translate_fallbacks") or []
        fb_note = (f" [WARNING: {len(fb)} line(s) kept in source language "
                   f"after translation retries]" if fb else "")
        job["message"] = (f"Done! ({len(cues)} subtitles, "
                          f"{detected}->{target_lang}, whisper "
                          f"{engine_info['model']}/{engine_info['device']}"
                          f"{cov_note}){note}{fb_note}")

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["message"] = str(e)


def start_job(job_id, source_lang, target_lang):
    t = threading.Thread(target=process_job, args=(job_id, source_lang, target_lang), daemon=True)
    t.start()
