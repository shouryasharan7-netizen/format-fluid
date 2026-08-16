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
        print("[FormatFluid] Loading Whisper model (tiny)...")
        _whisper_model = whisper.load_model("tiny")
        print("[FormatFluid] Whisper loaded.")
    return _whisper_model

def transcribe_audio(audio_path: str) -> dict:
    global _whisper_model
    if _whisper_model is None:
        get_whisper_model()
    
    print(f"[FormatFluid] Transcribing {audio_path}...")
    result = _whisper_model.transcribe(audio_path, language="en", fp16=False)
    
    # Free memory to prevent Railway OOM kills
    _whisper_model = None
    import gc
    gc.collect()
    
    return result

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Try to find a usable font
def find_font():
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
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
        "ffprobe", "-v", "error", 
        "-show_entries", "format=duration:stream=width,height,duration,r_frame_rate,codec_type",
        "-of", "json", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        data = json.loads(result.stdout)
    except Exception:
        raise ValueError("Failed to parse ffprobe output")
        
    streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise ValueError(f"No video stream found in {video_path}")
    
    info = streams[0]
    duration_str = info.get("duration")
    if duration_str is None:
        duration_str = data.get("format", {}).get("duration", "0")
        
    fps_str = info.get("r_frame_rate", "30/1")
    try:
        num, den = map(int, fps_str.split("/"))
        fps = num / den if den != 0 else 30
    except Exception:
        fps = 30
        
    return {
        "width": int(info.get("width", 1920)),
        "height": int(info.get("height", 1080)),
        "duration": float(duration_str) if duration_str else 0.0,
        "fps": fps
    }


def extract_audio(video_path: str, output_wav: str) -> str:
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        "-threads", "1",
        output_wav
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    return output_wav


def analyze_audio_energy(audio_path: str, duration: float, num_segments: int = 50) -> np.ndarray:
    energies = np.zeros(num_segments)
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 100:
        return energies
    try:
        import wave
        with wave.open(audio_path, 'rb') as wf:
            n_frames = wf.getnframes()
            audio_data = wf.readframes(n_frames)
            samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            samples_per_seg = max(1, len(samples) // num_segments)
            for i in range(num_segments):
                start_idx = i * samples_per_seg
                end_idx = start_idx + samples_per_seg if i < num_segments - 1 else len(samples)
                seg_samples = samples[start_idx:end_idx]
                if len(seg_samples) > 0:
                    energies[i] = np.sqrt(np.mean(np.square(seg_samples)))
    except Exception as e:
        print(f"Error analyzing audio energy: {e}")
    return energies


def detect_face_presence(video_path: str, duration: float, num_samples: int = 50) -> np.ndarray:
    if not _cv2_available:
        return np.zeros(num_samples)
    import cv2
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    scores = np.zeros(num_samples)
    
    if total_frames <= 0:
        return scores
        
    sample_interval = max(1, total_frames // num_samples)
    for i in range(total_frames):
        ret = cap.grab()
        if not ret:
            break
        if i % sample_interval == 0:
            idx = i // sample_interval
            if idx >= num_samples:
                break
            ret, frame = cap.retrieve()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Resize for drastically faster face detection
                h, w = gray.shape
                new_w = 480
                new_h = int(h * (new_w / w))
                gray_small = cv2.resize(gray, (new_w, new_h))
                faces = face_cascade.detectMultiScale(gray_small, 1.1, 4)
                if len(faces) > 0:
                    frame_area = new_w * new_h
                    max_face_area = max([fw * fh for (fx, fy, fw, fh) in faces])
                    scores[idx] = max_face_area / frame_area
    cap.release()
    return scores


def get_face_center(video_path: str, start_time: float, end_time: float) -> Tuple[Optional[float], Optional[float]]:
    if not _cv2_available:
        return None, None
    import cv2
    import tempfile
    
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    centers_x = []
    centers_y = []
    
    duration = end_time - start_time
    times_to_check = [start_time + duration * 0.25, start_time + duration * 0.5, start_time + duration * 0.75]
    
    for t in times_to_check:
        tmp_path = f"{video_path}_tmp_{t}.jpg"
        cmd = [
            "ffmpeg", "-y", "-ss", str(t), "-i", video_path, 
            "-vframes", "1", "-q:v", "2", tmp_path
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            if os.path.exists(tmp_path):
                img = cv2.imread(tmp_path)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    if len(faces) > 0:
                        largest = max(faces, key=lambda f: f[2] * f[3])
                        x, y, w, h = largest
                        centers_x.append(x + w / 2)
                        centers_y.append(y + h / 2)
                os.remove(tmp_path)
        except Exception:
            if os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except: pass
                
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
    if duration <= 0 or len(audio_energy) == 0:
        return []
        
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
    target_width: int = 720,
    target_height: int = 1280
) -> str:
    info = get_video_info(video_path)
    orig_w, orig_h = info["width"], info["height"]
    face_x, face_y = get_face_center(video_path, clip.start, clip.end)
    crop_h = orig_h
    crop_w = int(orig_h * (9 / 16))
    if crop_w > orig_w:
        crop_w = orig_w
        crop_h = int(orig_w * (16 / 9))
    if face_x is not None:
        crop_x = int(face_x - crop_w / 2)
        crop_x = max(0, min(crop_x, orig_w - crop_w))
    else:
        crop_x = (orig_w - crop_w) // 2
    crop_y = max(0, (orig_h - crop_h) // 2)
    duration = clip.end - clip.start
    cmd = [
        "ffmpeg", "-y", 
        "-ss", str(clip.start),
        "-i", video_path,
        "-t", str(clip.end - clip.start),
        "-vf", f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={target_width}:{target_height}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-c:a", "copy",
        "-threads", "1",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
        print(f"[FormatFluid] Crop failed, falling back to basic extraction for clip {clip.start}s")
        cmd_fallback = [
            "ffmpeg", "-y", 
            "-ss", str(clip.start),
            "-i", video_path,
            "-t", str(clip.end - clip.start),
            "-c:v", "copy",
            "-threads", "1",
            output_path
        ]
        result2 = subprocess.run(cmd_fallback, capture_output=True, text=True)
        if result2.returncode != 0 or not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
            if os.path.exists(video_path):
                shutil.copy(video_path, output_path)
    return output_path


def burn_captions(video_path: str, output_path: str, words: List[dict], caption_style: str = "minimalist") -> str:
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
        if caption_style == "mrbeast":
            style_params = f"fontcolor=yellow:fontsize={font_size + 20}:borderw=4:bordercolor=black:shadowx=3:shadowy=3:shadowcolor=black@0.7:x=(w-text_w)/2:y={y_pos - 40}"
        elif caption_style == "neon":
            style_params = f"fontcolor=cyan:fontsize={font_size + 10}:shadowx=2:shadowy=2:shadowcolor=magenta@0.8:borderw=2:bordercolor=magenta:x=(w-text_w)/2:y={y_pos}"
        else:
            style_params = f"fontcolor=white:fontsize={font_size}:box=1:boxcolor=black@0.6:boxborderw=10:x=(w-text_w)/2:y={y_pos}"
            
        dt = (
            f"drawtext=fontfile={FONT_PATH}:"
            f"text='{safe_text}':"
            f"{style_params}:"
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
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "copy",
        "-threads", "1",
        "-movflags", "+faststart",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
        if os.path.exists(video_path):
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


def process_video(video_path: str, job_id: str, progress_cb=None, caption_style="minimalist") -> dict:
    def update_progress(stage, status, pct):
        if progress_cb:
            progress_cb(stage, status, pct)

    update_progress(1, "Validating and decoding video file...", 5)
    print(f"[{job_id}] Starting processing for: {video_path}")
    job_output = OUTPUT_DIR / job_id
    job_output.mkdir(exist_ok=True)
    info = get_video_info(video_path)
    print(f"[{job_id}] Video: {info['width']}x{info['height']}, {info['duration']:.1f}s")
    
    update_progress(1, "Extracting audio track...", 15)
    audio_path = str(job_output / "audio.wav")
    extract_audio(video_path, audio_path)
    
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 100:
        print(f"[{job_id}] Audio extraction failed or no audio track.")
        transcript = {}
    else:
        update_progress(2, "Transcribing speech for smart detection...", 30)
        transcript = transcribe_audio(audio_path)
    
    print(f"[{job_id}] Transcription done. Segments: {len(transcript.get('segments', []))}")
    update_progress(2, "Running AI smart-detection for speaker focus and high-energy windows...", 45)
    audio_energy = analyze_audio_energy(audio_path, info["duration"])
    face_scores = detect_face_presence(video_path, info["duration"], num_samples=len(audio_energy))
    clip_duration = min(15.0, info["duration"] / 3)
    clips = find_viral_moments(
        info["duration"], transcript, audio_energy, face_scores,
        num_clips=3, clip_duration=clip_duration
    )
    print(f"[{job_id}] Found {len(clips)} clips")
    
    results = []
    for i, clip in enumerate(clips):
        base_pct = 50 + (i / len(clips)) * 40
        print(f"[{job_id}] Processing clip {i+1}/{len(clips)}: {clip.start:.1f}s - {clip.end:.1f}s")
        
        update_progress(3, f"Auto-reframing to 9:16 vertical crop (Clip {i+1}/{len(clips)})...", int(base_pct))
        raw_path = str(job_output / f"clip_{i}_raw.mp4")
        extract_vertical_clip(video_path, clip, raw_path)
        
        clip_words = []
        if "segments" in transcript:
            for seg in transcript["segments"]:
                if "words" in seg:
                    for word in seg["words"]:
                        if clip.start <= word["start"] <= clip.end:
                            clip_words.append(word)
                            
        update_progress(4, f"Burning word-level synced captions (Clip {i+1}/{len(clips)})...", int(base_pct + 5))
        final_path = str(job_output / f"clip_{i}.mp4")
        burn_captions(raw_path, final_path, clip_words, caption_style=caption_style)
        
        update_progress(5, f"Generating platform-specific copy (Clip {i+1}/{len(clips)})...", int(base_pct + 10))
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