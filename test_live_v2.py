"""Live Railway verification: upload a clip, poll, download, extract QC frame."""
import json
import subprocess
import sys
import time
import urllib.request

BASE = "https://video-translator-production-8218.up.railway.app"
CLIP = sys.argv[1] if len(sys.argv) > 1 else r"jobs\cpu_test\input.mp4"
TARGET = sys.argv[2] if len(sys.argv) > 2 else "es"


def api(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.load(r)


print("health:", json.dumps(api("/health")))

# multipart upload
import mimetypes, uuid  # noqa: E402
boundary = uuid.uuid4().hex
body = b""
for name, val in (("source_lang", "auto"), ("target_lang", TARGET)):
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{val}\r\n").encode()
data = open(CLIP, "rb").read()
body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"clip.mp4\"\r\n"
         f"Content-Type: video/mp4\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
req = urllib.request.Request(BASE + "/translate", data=body, headers={
    "Content-Type": f"multipart/form-data; boundary={boundary}"})
job = json.load(urllib.request.urlopen(req, timeout=120))
job_id = job["job_id"]
print("job:", job_id)

t0 = time.time()
while True:
    s = api(f"/status/{job_id}")
    print(f"  [{time.time()-t0:5.0f}s] {s['status']} {s.get('progress')}% - {s.get('message','')[:100]}")
    if s["status"] in ("done", "error"):
        break
    time.sleep(10)

if s["status"] != "done":
    sys.exit("LIVE JOB FAILED: " + s.get("message", ""))

print("engine:", s.get("engine"))
srt = api(f"/srt/{job_id}")
print("SRT first lines:\n" + "\n".join(srt["content"].splitlines()[:12]))

out = r"jobs\live_output.mp4"
urllib.request.urlretrieve(BASE + f"/download/{job_id}", out)
print("downloaded:", out)
subprocess.run(["ffmpeg", "-y", "-ss", "3", "-i", out, "-frames:v", "1",
                r"jobs\qc_live.png"], check=True, capture_output=True)
subprocess.run(["ffmpeg", "-y", "-ss", "12", "-i", out, "-frames:v", "1",
                r"jobs\qc_live2.png"], check=True, capture_output=True)
print("frames: jobs/qc_live.png jobs/qc_live2.png")
