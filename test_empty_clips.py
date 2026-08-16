import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Clip:
    start: float
    end: float
    score: float
    text: str

def find_viral_moments(duration, transcript, audio_energy, face_scores, num_clips=3, clip_duration=15.0):
    num_segments = len(audio_energy)
    if num_segments == 0: return []
    segment_duration = duration / num_segments
    audio_min, audio_max = audio_energy.min(), audio_energy.max()
    audio_norm = (audio_energy - audio_min) / (audio_max - audio_min + 1e-8)
    face_max = face_scores.max()
    face_norm = face_scores / (face_max + 1e-8) if face_max > 0 else np.zeros_like(face_scores)
    word_counts = np.zeros(num_segments)
    word_norm = word_counts / (word_counts.max() + 1e-8)
    combined = 0.5 * audio_norm + 0.5 * word_norm
    clips = []
    min_gap = clip_duration * 0.8
    for _ in range(num_clips):
        if len(combined) == 0: break
        best_idx = int(np.argmax(combined))
        best_time = best_idx * segment_duration
        start = max(0.0, best_time - clip_duration / 2)
        end = min(duration, start + clip_duration)
        start = max(0.0, end - clip_duration)
        clips.append(Clip(start=start, end=end, score=float(combined[best_idx]), text=""))
        start_idx = max(0, int((best_time - min_gap) / segment_duration))
        end_idx = min(num_segments, int((best_time + min_gap) / segment_duration))
        combined[start_idx:end_idx] = 0.0
    return clips

print(find_viral_moments(5.0, {}, np.zeros(100), np.zeros(100)))
