# Format Fluid

**[Live Demo](https://format-fluid-production.up.railway.app)**

> Turn one long-form video into platform-ready short clips — with auto-captions, smart reframing, and native copy for every platform.

## The Problem

Creators spend 40% of their week repurposing content. A 20-min YouTube video becomes 5 TikToks, 3 Shorts, a Reel, and a Twitter thread — 6 hours of manual work.

## The Solution

Drop a video. Get vertical clips, synced captions, and platform-native copy in under 2 minutes.

## Demo

![Format Fluid Demo](demo.gif)

## How It Works

| Stage | Tech | Signal |
|-------|------|--------|
| Transcribe | OpenAI Whisper | Word-level timestamps |
| Score | Custom heuristic | 40% audio energy + 30% face presence + 30% word density |
| Reframe | FFmpeg + OpenCV | Smart 9:16 crop centered on faces |
| Caption | FFmpeg drawtext | 3-second chunked captions with black box |
| Copy | Rule-based gen | Platform-native hooks per network |

## Tech Stack

- Python + Flask
- OpenAI Whisper (base)
- OpenCV (face detection)
- FFmpeg (video processing)
- Vanilla JS

## Run Locally

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```
