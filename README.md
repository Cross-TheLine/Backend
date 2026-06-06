# Cross The Line Backend

Tennis ball tracking and bounce preprocessing backend for segmented tennis videos.

This backend is responsible for the pipeline before in/out judgment:

```text
recording video + pressed time
-> clip extraction
-> ball tracking
-> bounce detection
-> JSON payload for in/out judgment
```

## Environment

- Python 3.11+
- Install non-Torch packages:

```powershell
pip install -r requirements.txt
```

- Install CUDA PyTorch for NVIDIA GPU inference:

```powershell
pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Required Weights

```text
weights/tracknet_pretrained.pt
```

## Folders

- `input/slow_inputs_seg/`: segmented input videos.
- `input/slow_inputs_unity/`: Unity input videos.
- `output/`: generated result folders and archives.
- `src/ball_tracking/`: TrackNet model, heatmap post-processing, ball trajectory tracking.
- `src/bounce_detection/`: bounce detection from tracked coordinates.
- `src/inout_judgement/`: in/out judgment placeholder/package.
- `src/line_detection/`: AprilTag and court-line detection helpers.
- `src/api/`: FastAPI server.

## Run API Server

```powershell
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

Interactive docs:

```text
http://localhost:8000/docs
```

## MVP App Flow

Use this if the app already records a full video file and sends the button press time:

```text
POST /judge-preprocess
GET  /jobs/{job_id}
```

Request:

```json
{
  "recording_path": "C:/path/to/full_recording.mp4",
  "pressed_at_sec": 153.42,
  "lookback_sec": 4.0,
  "render_video": true
}
```

Initial response:

```json
{
  "job_id": "uuid",
  "status": "pending"
}
```

Poll `GET /jobs/{job_id}` until `status` is `done`.

Done response includes:

```json
{
  "status": "done",
  "result": "bounce_detected",
  "confidence": 0.8,
  "primary_bounce": {
    "frame_index": 38,
    "clip_time_sec": 1.267,
    "recording_time_sec": 151.267,
    "x": 1391.96,
    "y": 564.65,
    "score": 49.18
  },
  "bounces": [],
  "inout_input": {
    "fps": 30.0,
    "frame_width": 1920,
    "frame_height": 1080,
    "clip_start_sec": 150.0,
    "bounces": [],
    "track_csv": "output/api_sessions/.../track.csv",
    "bounces_csv": "output/api_sessions/.../bounces.csv"
  },
  "artifacts": {
    "clip": "/files/.../clip.mp4",
    "track_csv": "/files/.../track.csv",
    "bounces_csv": "/files/.../bounces.csv",
    "result_video": "/files/.../result.avi"
  }
}
```

`inout_input` is the JSON block intended for the in/out judgment module.

## Session-Based Flow

Use this if the app wants backend-managed session records:

```text
POST /sessions/start
POST /sessions/{session_id}/line-status
POST /sessions/{session_id}/record/start
POST /sessions/{session_id}/record/upload
POST /sessions/{session_id}/record/path
POST /sessions/{session_id}/judge
GET  /jobs/{job_id}
POST /sessions/{session_id}/save
POST /sessions/{session_id}/finish
```

Generated files are stored under:

```text
output/api_sessions/{session_id}/jobs/{job_id}/
```

Each job writes:

- `clip.mp4`
- `track.csv`
- `bounces.csv`
- `result.avi` when `render_video` is true
- `job.json`

## CLI: Y-Reversal Bounce Detection

```powershell
python -m src.bounce_detection.detect_bounces_from_track_csv --video_path ".\input\slow_inputs_seg\3대 작은마커\try5\try5_seg08_slow_0_5x.avi" --output_root .\output\output_y_reversal_bounces --device cuda
```
