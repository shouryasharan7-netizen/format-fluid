from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import uuid
from pathlib import Path
from processor import process_video, UPLOAD_DIR, OUTPUT_DIR

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

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
    
    try:
        result = process_video(str(upload_path), job_id)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": repr(e)}), 500

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