from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import uuid
from pathlib import Path
from processor import process_video, UPLOAD_DIR, OUTPUT_DIR

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

import threading

JOB_STATUSES = {}

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
    
    JOB_STATUSES[job_id] = {
        "stage": 1, 
        "status": "Initializing...", 
        "progress": 0,
        "result": None,
        "error": None
    }
    
    def run_job(path, jid, style):
        def progress_cb(stage, status, pct):
            JOB_STATUSES[jid]["stage"] = stage
            JOB_STATUSES[jid]["status"] = status
            JOB_STATUSES[jid]["progress"] = pct
            
        try:
            result = process_video(path, jid, progress_cb=progress_cb, caption_style=style)
            JOB_STATUSES[jid]["result"] = result
            JOB_STATUSES[jid]["progress"] = 100
        except Exception as e:
            import traceback
            traceback.print_exc()
            JOB_STATUSES[jid]["error"] = repr(e)
            
    threading.Thread(target=run_job, args=(str(upload_path), job_id, caption_style)).start()
    
    return jsonify({"job_id": job_id})

@app.route("/status/<job_id>")
def status(job_id):
    if job_id not in JOB_STATUSES:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(JOB_STATUSES[job_id])

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