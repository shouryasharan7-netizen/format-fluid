"""
Format Fluid - Core Video Processor
Turns long-form video into short-form clips with auto-captions
"""
import os
import json
import subprocess
import shutil
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Lazy-load Whisper
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print("[FormatFluid] Loading Whisper model (base)...")
        _whisper_model = whisper.load_model("base")
        print("[FormatFluid] Whisper loaded.")
    return _whisper_model

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Try to find a usable font
def find_font():
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    try:
        result = subprocess.run(["find", "/usr/share/fonts", "/System/Library/Fonts", "/Library/Fonts", "-name", "*.ttf"], capture_output=True, text=True)
        lines = result.stdout.strip().split("\n")
        if lines and lines[0]:
            return lines[0]
    except:
        pass
    return "Arial"

FONT_PATH = find_font()
print(f"[FormatFluid] Using font: {FONT_PATH}")

# Bulletproof cv2 check — test CascadeClassifier specifically
_cv2_available = False
try:
    import cv2
    _test_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    _test_cascade = cv2.CascadeClassifier(_test_path)
    if _test_cascade.empty():
        print("[FormatFluid] OpenCV loaded but cascade classifier is empty. Disabling face detection.")
    else:
        _cv2_available = True
        print("[FormatFluid] OpenCV + face detection loaded.")
except Exception as e:
    print(f"[FormatFluid] OpenCV not available ({e}), using audio-only detection.")


@dataclass
class Clip:
    start: float
    end: float
    score: float
    text: str
    output_path: str = ""


def get_video_info(video_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration,r_frame_rate",
        "-of", "json", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        cmd2 = [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=width,height,duration,r_frame_rate",
            "-of", "json", video_path
        ]
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        data2 = json.loads(result2.stdout)
        streams = data2.get("streams", [])
    if not streams:
        raise ValueError(f"No video stream found in {video_path}")
    info = streams[0]
    fps_str = info.get("r_frame_rate", "30/1")
    try:
        num, den = map(int, fps_str.split("/"))
        fps = num / den if den != 0 else 30
    except Exception:
        fps = 30
    return {
        "width": int(info.get("width", 1920)),
        "height": int(info.get("height", 1080)),
        "duration": float(info.get("duration", 0)),
        "fps": fps
    }


def extract_audio(video_path: str, output_wav: str) -> str:
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        output_wav
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    return output_wav


def transcribe_audio(audio_path: str) -> dict:
    model = get_whisper_model()
    result = model.transcribe(
        audio_path, 
        word_timestamps=True,
        condition_on_previous_text=False
    )
    return result


def analyze_audio_energy(video_path: str, duration: float, num_segments: int = 100) -> np.ndarray:
    segment_duration = duration / num_segments
    energies = []
    for i in range(num_segments):
        start = i * segment_duration
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-ss", str(start), "-t", str(segment_duration),
            "-af", "ebur128=peak=true",
            "-f", "null", "-"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stderr
        loudness = -70.0
        for line in output.split("\n"):
            if "Integrated loudness:" in line and "I:" in line:
                try:
                    loudness = float(line.split("I:")[1].split("LUFS")[0].strip())
                except Exception:
                    pass
        energies.append(loudness)
    return np.array(energies)


def detect_face_presence(video_path: str, duration: float, num_samples: int = 50) -> np.ndarray:
    if not _cv2_available:
        return np.zeros(num_samples)
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    scores = np.zeros(num_samples)
    sample_interval = total_frames / num_samples if total_frames > 0 else 1
    for i in range(num_samples):
        frame_idx = int(i * sample_interval)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            frame_area = frame.shape[0] * frame.shape[1]
            max_face_area = max([w * h for (x, y, w, h) in faces])
            scores[i] = max_face_area / frame_area
    cap.release()
    return scores


def get_face_center(video_path: str, start_time: float, end_time: float) -> Tuple[Optional[float], Optional[float]]:
    if not _cv2_available:
        return None, None
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    centers_x = []
    centers_y = []
    step = max(1, (end_frame - start_frame) // 10)
    for frame_idx in range(start_frame, end_frame, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            largest = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest
            centers_x.append(x + w / 2)
            centers_y.append(y + h / 2)
    cap.release()
    if centers_x:
        return np.median(centers_x), np.median(centers_y)
    return None, None


def find_viral_moments(
    duration: float,
    transcript: dict,
    audio_energy: np.ndarray,
    face_scores: np.ndarray,
    num_clips: int = 5,
    clip_duration: float = 15.0
) -> List[Clip]:
    num_segments = len(audio_energy)
    segment_duration = duration / num_segments
    audio_min, audio_max = audio_energy.min(), audio_energy.max()
    audio_norm = (audio_energy - audio_min) / (audio_max - audio_min + 1e-8)
    face_max = face_scores.max()
    face_norm = face_scores / (face_max + 1e-8) if face_max > 0 else np.zeros_like(face_scores)
    word_counts = np.zeros(num_segments)
    if "segments" in transcript:
        for seg in transcript["segments"]:
            if "words" in seg:
                for word in seg["words"]:
                    idx = min(int(word["start"] / segment_duration), num_segments - 1)
                    word_counts[idx] += 1
    word_norm = word_counts / (word_counts.max() + 1e-8)
    if not _cv2_available:
        combined = 0.5 * audio_norm + 0.5 * word_norm
    else:
        combined = 0.4 * audio_norm + 0.3 * face_norm + 0.3 * word_norm
    clips = []
    min_gap = clip_duration * 0.8
    for _ in range(num_clips):
        if len(combined) == 0:
            break
        best_idx = int(np.argmax(combined))
        best_time = best_idx * segment_duration
        start = max(0.0, best_time - clip_duration / 2)
        end = min(duration, start + clip_duration)
        start = max(0.0, end - clip_duration)
        clip_text = ""
        if "segments" in transcript:
            words = []
            for seg in transcript["segments"]:
                if "words" in seg:
                    for word in seg["words"]:
                        if start <= word["start"] <= end:
                            words.append(word["word"])
            clip_text = "".join(words).strip()
        clips.append(Clip(start=start, end=end, score=float(combined[best_idx]), text=clip_text))
        start_idx = max(0, int((best_time - min_gap) / segment_duration))
        end_idx = min(num_segments, int((best_time + min_gap) / segment_duration))
        combined[start_idx:end_idx] = 0.0
    return clips


def extract_vertical_clip(
    video_path: str,
    clip: Clip,
    output_path: str,
    target_width: int = 1080,
    target_height: int = 1920
) -> str:
    info = get_video_info(video_path)
    orig_w, orig_h = info["width"], info["height"]
    face_x, face_y = get_face_center(video_path, clip.start, clip.end)
    crop_height = orig_h
    crop_width = int(orig_h * (9 / 16))
    if crop_width > orig_w:
        crop_width = orig_w
        crop_height = int(orig_w * (16 / 9))
    if face_x is not None:
        crop_x = int(face_x - crop_width / 2)
        crop_x = max(0, min(crop_x, orig_w - crop_width))
    else:
        crop_x = (orig_w - crop_width) // 2
    crop_y = max(0, (orig_h - crop_height) // 2)
    duration = clip.end - clip.start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(clip.start),
        "-t", str(duration),
        "-i", video_path,
        "-vf", f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y},scale={target_width}:{target_height}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        output_path
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    return output_path


def burn_captions(video_path: str, output_path: str, words: List[dict]) -> str:
    if not words:
        shutil.copy(video_path, output_path)
        return output_path
    info = get_video_info(video_path)
    w, h = info["width"], info["height"]
    font_size = int(h * 0.055)
    y_pos = int(h * 0.82)
    chunk_size = 3.0
    chunks = []
    current_chunk = []
    chunk_start = words[0]["start"] if words else 0.0
    for word in words:
        if word["start"] - chunk_start > chunk_size:
            if current_chunk:
                end_time = current_chunk[-1].get("end", current_chunk[-1]["start"] + 1.0)
                chunk_text = " ".join([w["word"].strip() for w in current_chunk])
                chunks.append((chunk_start, end_time, chunk_text))
            current_chunk = [word]
            chunk_start = word["start"]
        else:
            current_chunk.append(word)
    if current_chunk:
        end_time = current_chunk[-1].get("end", current_chunk[-1]["start"] + 1.0)
        chunk_text = " ".join([w["word"].strip() for w in current_chunk])
        chunks.append((chunk_start, end_time, chunk_text))
    drawtexts = []
    for start, end, text in chunks:
        safe_text = text.replace("\\", "\\\\").replace("'", "\\'") \
                        .replace(":", "\\:").replace("[", "\\[") \
                        .replace("]", "\\]").replace("=", "\\=") \
                        .replace(",", "\\,")
        if not safe_text.strip():
            continue
        dt = (
            f"drawtext=fontfile={FONT_PATH}:"
            f"text='{safe_text}':"
            f"fontcolor=white:fontsize={font_size}:"
            f"x=(w-text_w)/2:y={y_pos}:"
            f"box=1:boxcolor=black@0.6:boxborderw=10:"
            f"enable='between(t\\,{start}\\,{end})'"
        )
        drawtexts.append(dt)
    if not drawtexts:
        shutil.copy(video_path, output_path)
        return output_path
    vf = ",".join(drawtexts)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "copy",
        "-movflags", "+faststart",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
        shutil.copy(video_path, output_path)
    return output_path


def generate_platform_copy(clip_text: str) -> dict:
    text = clip_text[:200] if len(clip_text) > 200 else clip_text
    words = text.split()
    hook = " ".join(words[:10]) if len(words) > 10 else text
    return {
        "tiktok": f"{hook}... \n\n#viral #fyp #trending #content",
        "youtube_shorts": f"{hook}... \n\n#shorts #viral",
        "instagram_reels": f"{hook}... \n\n.\n.\n.\n#reels #instagood #contentcreator",
        "twitter": f"{hook}... \n\nWhat do you think? \n\n#content #creator",
        "linkedin": f"Insight: {hook}... \n\nWhat's your take on this? \n\n#contentstrategy #creator"
    }


def process_video(video_path: str, job_id: str) -> dict:
    print(f"[{job_id}] Starting processing for: {video_path}")
    job_output = OUTPUT_DIR / job_id
    job_output.mkdir(exist_ok=True)
    info = get_video_info(video_path)
    print(f"[{job_id}] Video: {info['width']}x{info['height']}, {info['duration']:.1f}s")
    audio_path = str(job_output / "audio.wav")
    extract_audio(video_path, audio_path)
    transcript = transcribe_audio(audio_path)
    print(f"[{job_id}] Transcription done. Segments: {len(transcript.get('segments', []))}")
    audio_energy = analyze_audio_energy(video_path, info["duration"])
    face_scores = detect_face_presence(video_path, info["duration"])
    clip_duration = min(20.0, info["duration"] / 3)
    clips = find_viral_moments(
        info["duration"], transcript, audio_energy, face_scores,
        num_clips=5, clip_duration=clip_duration
    )
    print(f"[{job_id}] Found {len(clips)} clips")
    results = []
    for i, clip in enumerate(clips):
        print(f"[{job_id}] Processing clip {i+1}/{len(clips)}: {clip.start:.1f}s - {clip.end:.1f}s")
        raw_path = str(job_output / f"clip_{i}_raw.mp4")
        extract_vertical_clip(video_path, clip, raw_path)
        clip_words = []
        if "segments" in transcript:
            for seg in transcript["segments"]:
                if "words" in seg:
                    for word in seg["words"]:
                        if clip.start <= word["start"] <= clip.end:
                            clip_words.append(word)
        final_path = str(job_output / f"clip_{i}.mp4")
        burn_captions(raw_path, final_path, clip_words)
        copy = generate_platform_copy(clip.text)
        results.append({
            "clip_index": i,
            "start": round(clip.start, 2),
            "end": round(clip.end, 2),
            "duration": round(clip.end - clip.start, 2),
            "score": round(clip.score, 3),
            "text_preview": clip.text[:100] + "..." if len(clip.text) > 100 else clip.text,
            "filename": f"clip_{i}.mp4",
            "platform_copy": copy
        })
    if os.path.exists(audio_path):
        os.remove(audio_path)
    for f in job_output.glob("*_raw.mp4"):
        os.remove(f)
    print(f"[{job_id}] Done! Generated {len(results)} clips.")
    return {
        "job_id": job_id,
        "original": {
            "duration": info["duration"],
            "width": info["width"],
            "height": info["height"]
        },
        "clips": results,
        "output_dir": str(job_output)
    }