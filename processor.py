import whisper
import numpy as np
import subprocess
import threading
import os
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
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def load_audio_with_ffmpeg(ffmpeg_exe, input_path, sr=16000):
    """Extract audio using our known ffmpeg path, return float32 numpy array."""
    cmd = [
        ffmpeg_exe, "-nostdin", "-threads", "0",
        "-i", input_path,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr),
        "-"
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0


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

        model = whisper.load_model("small")

        jobs[job_id]["progress"] = 25
        jobs[job_id]["message"] = "Transcribing audio..."

        audio = load_audio_with_ffmpeg(ffmpeg_exe, input_path)

        # When target is English, use Whisper's native translation (far more accurate)
        if target_lang == "en":
            kwargs = {"task": "translate"}
            if source_lang != "auto":
                kwargs["language"] = source_lang
            result = model.transcribe(audio, **kwargs)
            use_whisper_translation = True
        else:
            kwargs = {}
            if source_lang != "auto":
                kwargs["language"] = source_lang
            result = model.transcribe(audio, **kwargs)
            use_whisper_translation = False

        jobs[job_id]["progress"] = 50
        jobs[job_id]["message"] = "Translating..."

        srt_lines = []
        for i, seg in enumerate(result["segments"], 1):
            text = seg["text"].strip()
            if not use_whisper_translation:
                text = GoogleTranslator(source="auto", target=target_lang).translate(text)
            srt_lines.append(str(i))
            srt_lines.append(f'{fmt_time(seg["start"])} --> {fmt_time(seg["end"])}')
            srt_lines.append(text)
            srt_lines.append("")

        srt_content = "\n".join(srt_lines)
        srt_path = os.path.join(JOBS_DIR, job_id, "subtitles.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        jobs[job_id]["progress"] = 75
        jobs[job_id]["message"] = "Burning subtitles..."

        output_path = os.path.join(JOBS_DIR, job_id, "output.mp4")

        # Escape path for ffmpeg subtitles filter: backslashes → forward slashes, colons escaped
        escaped = srt_path.replace("\\", "/").replace(":", "\\\\:")

        # force_style commas must be escaped with \ when not going through a shell
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
            # Store last 300 chars of stderr for debugging even on success
            jobs[job_id]["message"] = "Done! | " + result2.stderr[-300:].replace("\n", " ")

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["message"] = str(e)


def start_job(job_id, source_lang, target_lang):
    t = threading.Thread(target=process_job, args=(job_id, source_lang, target_lang), daemon=True)
    t.start()
