import uuid
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import yt_dlp
import processor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(BASE_DIR, "jobs")


@asynccontextmanager
async def lifespan(app):
    os.makedirs(JOBS_DIR, exist_ok=True)
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/translate")
async def translate(
    file: UploadFile = File(None),
    url: str = Form(None),
    source_lang: str = Form(...),
    target_lang: str = Form(...),
):
    job_id = uuid.uuid4().hex[:8]
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir)

    processor.jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Queued",
        "output_path": None,
    }

    try:
        if url:
            ydl_opts = {
                "outtmpl": os.path.join(job_dir, "input.%(ext)s").replace("\\", "/"),
                "format": "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "cookiesfrombrowser": ("chrome",),
                "quiet": True,
                "no_warnings": False,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        elif file:
            contents = await file.read()
            input_path = os.path.join(job_dir, "input.mp4")
            with open(input_path, "wb") as f:
                f.write(contents)
        else:
            return JSONResponse({"error": "No file or URL provided"}, status_code=400)
    except Exception as e:
        import shutil
        shutil.rmtree(job_dir, ignore_errors=True)
        del processor.jobs[job_id]
        return JSONResponse({"error": str(e)}, status_code=500)

    processor.start_job(job_id, source_lang, target_lang)
    return JSONResponse({"job_id": job_id})


@app.get("/status/{job_id}")
async def status(job_id: str):
    if job_id not in processor.jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return JSONResponse(processor.jobs[job_id])


@app.get("/download/{job_id}")
async def download(job_id: str):
    job = processor.jobs.get(job_id)
    if not job or job["status"] != "done":
        return JSONResponse({"error": "Job not ready"}, status_code=404)
    return FileResponse(job["output_path"], media_type="video/mp4", filename="translated.mp4")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
