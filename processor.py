import subprocess
import threading
import os
import gc
from deep_translator import GoogleTranslator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(BASE_DIR, "jobs")

jobs = {}


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


def process_job(job_id, source_lang, target_lang):
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 5
        jobs[job_id]["message"] = "Loading model..."

        ffmpeg_exe = _get_ffmpeg()

        job_dir = os.path.join(JOBS_DIR, job_id)
        input_path = None
        for f in os.listdir(job_dir):
            if f.startswith("input"):
                input_path = os.path.join(job_dir, f)
                break

        if input_path is None:
            raise FileNotFoundError("No input file found in job directory")

        from faster_whisper import WhisperModel
        # tiny int8: ~150MB peak RAM — fits Railway 512MB with headroom
        model = WhisperModel("tiny", device="cpu", compute_type="int8")

        jobs[job_id]["progress"] = 20
        jobs[job_id]["message"] = "Transcribing audio..."

        kwargs = {}
        if source_lang != "auto":
            kwargs["language"] = source_lang

        if target_lang == "en":
            kwargs["task"] = "translate"

        segments_gen, _ = model.transcribe(input_path, **kwargs)
        segments = list(segments_gen)

        # Free model memory immediately before translation
        del model
        gc.collect()

        jobs[job_id]["progress"] = 50
        jobs[job_id]["message"] = "Translating..."

        srt_lines = []
        for i, seg in enumerate(segments, 1):
            text = seg.text.strip()
            if target_lang != "en":
                text = GoogleTranslator(source="auto", target=target_lang).translate(text)
            srt_lines.append(str(i))
            srt_lines.append(f"{fmt_time(seg.start)} --> {fmt_time(seg.end)}")
            srt_lines.append(text)
            srt_lines.append("")

        srt_content = "\n".join(srt_lines)
        srt_path = os.path.join(JOBS_DIR, job_id, "subtitles.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        jobs[job_id]["progress"] = 75
        jobs[job_id]["message"] = "Burning subtitles..."

        output_path = os.path.join(JOBS_DIR, job_id, "output.mp4")
        escaped = srt_path.replace("\\", "/").replace(":", "\\\\:")
        style = "FontSize=24\\,PrimaryColour=&H00FFFFFF\\,OutlineColour=&H00000000\\,Outline=2\\,Shadow=1"

        cmd = [
            ffmpeg_exe, "-y",
            "-i", input_path,
            "-vf", f"subtitles={escaped}:force_style={style}",
            "-c:a", "copy",
            output_path
        ]

        result2 = subprocess.run(cmd, capture_output=True, text=True, errors="replace")

        if result2.returncode != 0:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["message"] = result2.stderr[-800:]
        else:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["output_path"] = output_path
            jobs[job_id]["message"] = "Done!"

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["message"] = str(e)


def start_job(job_id, source_lang, target_lang):
    t = threading.Thread(target=process_job, args=(job_id, source_lang, target_lang), daemon=True)
    t.start()
