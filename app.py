from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import uuid
from pathlib import Path
from processor import process_video, UPLOAD_DIR, OUTPUT_DIR

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

import threading
import json

JOB_STATUSES = {}

def update_job_status(jid, stage, status, pct, result=None, error=None):
    data = {
        "stage": stage,
        "status": status,
        "progress": pct,
        "result": result,
        "error": error
    }
    JOB_STATUSES[jid] = data
    try:
        with open(OUTPUT_DIR / f"status_{jid}.json", "w") as f:
            json.dump(data, f)
    except Exception:
        pass

def get_job_status(jid):
    if jid in JOB_STATUSES:
        return JOB_STATUSES[jid]
    status_file = OUTPUT_DIR / f"status_{jid}.json"
    if status_file.exists():
        try:
            with open(status_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400
    
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    
    job_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix or ".mp4"
    upload_path = UPLOAD_DIR / f"{job_id}{ext}"
    file.save(str(upload_path))
    
    caption_style = request.form.get("caption_style", "minimalist")
    
    update_job_status(job_id, 1, "Initializing...", 0)
    
    def run_job(path, jid, style):
        def progress_cb(stage, status, pct):
            update_job_status(jid, stage, status, pct)
            
        try:
            result = process_video(path, jid, progress_cb=progress_cb, caption_style=style)
            update_job_status(jid, 5, "Done!", 100, result=result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            update_job_status(jid, 5, "Error", 100, error=repr(e))
            
    threading.Thread(target=run_job, args=(str(upload_path), job_id, caption_style)).start()
    
    return jsonify({"job_id": job_id})

@app.route("/status/<job_id>")
def status(job_id):
    status_data = get_job_status(job_id)
    if not status_data:
        return jsonify({"error": "Job not found or Server restarted."}), 404
    return jsonify(status_data)

@app.route("/download/<job_id>/<filename>")
def download(job_id, filename):
    directory = OUTPUT_DIR / job_id
    return send_from_directory(str(directory), filename, as_attachment=True)

@app.route("/clips/<job_id>/<filename>")
def serve_clip(job_id, filename):
    directory = OUTPUT_DIR / job_id
    return send_from_directory(str(directory), filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)