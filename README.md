# Cross The Line Backend

Tennis ball tracking, bounce detection, and in/out judgment backend for segmented tennis videos.

This backend is responsible for this pipeline:

```text
recording video + pressed time
-> clip extraction
-> ball tracking
-> bounce detection
-> optional in/out judgment with court-line JSON
-> JSON and overlay artifacts
```

## Environment

- Python 3.11+
- Install packages:

```powershell
pip install -r requirements.txt
```

- If you use an NVIDIA GPU, replace the default PyTorch install with the CUDA wheel:

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
- `src/inout_judgement/`: in/out judgment and combined overlay helpers.
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

Runtime readiness:

```text
GET /health
```

`judge_ready` is true only when both `torch` and `weights/tracknet_pretrained.pt` are available.

## MVP App Flow

Use this if the app already records a full video file and sends the button press time:

```text
POST /judge-preprocess
GET  /jobs/{job_id}
GET  /jobs/{job_id}/result
```

Request:

```json
{
  "recording_path": "C:/path/to/full_recording.mp4",
  "use_video_end": true,
  "lookback_sec": 2.0,
  "render_video": true,
  "court_config_path": "C:/path/to/view2_lines.json",
  "config_image": null,
  "config_index": 0,
  "render_inout_video": true
}
```

If the app stops recording when the user presses the judgment button, omit
`pressed_at_sec` or set `use_video_end=true`. The backend reads `video_duration_sec`
from the uploaded video and judges the last `lookback_sec` seconds:

```text
pressed_at_sec = video_duration_sec - end_offset_sec
clip_start_sec = max(0, pressed_at_sec - lookback_sec)
```

With the default `lookback_sec=2.0`, videos shorter than 2 seconds are judged from
the beginning of the uploaded video.

Use `end_offset_sec` only if the app keeps recording briefly after the button press.
For example, if recording stops 0.5 seconds after the press, send
`"end_offset_sec": 0.5`.

For manually synchronized clients, `pressed_at_sec` is still supported. If it exceeds
the uploaded video duration, the backend clamps it to the video end.

Initial response:

```json
{
  "job_id": "uuid",
  "status": "pending"
}
```

Poll `GET /jobs/{job_id}` until `status` is `done`, or use `GET /jobs/{job_id}/result` for the frontend-stable response shape.

Frontend result response:

```json
{
  "job_id": "uuid",
  "session_id": "uuid",
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
  "primary_decision": {
    "frame_index": 38,
    "clip_time_sec": 1.267,
    "recording_time_sec": 151.267,
    "x": 1391.96,
    "y": 564.65,
    "decision": "IN",
    "decision_reason": "inside_polygon",
    "boundary_distance_px": 42.5,
    "signed_distance_px": null
  },
  "bounces": [],
  "decisions": [],
  "artifacts": {
    "clip": "/files/.../clip.mp4",
    "track_csv": "/files/.../track.csv",
    "bounces_csv": "/files/.../bounces.csv",
    "result_video": "/files/.../result.avi",
    "inout_csv": "/files/.../inout_judged.csv",
    "inout_overlay_video": "/files/.../inout_overlay.avi"
  },
  "error": null
}
```

If `court_config_path` is omitted, the job still returns ball tracking and bounce detection only.

## Session-Based Flow

Use this if the app wants backend-managed session records:

```text
POST /sessions/start
POST /sessions/{session_id}/line-status
POST /sessions/{session_id}/court-config/detect
POST /sessions/{session_id}/court-config/upload
POST /sessions/{session_id}/court-config/path
POST /sessions/{session_id}/record/start
POST /sessions/{session_id}/record/upload
POST /sessions/{session_id}/record/path
POST /sessions/{session_id}/judge
GET  /jobs/{job_id}
GET  /jobs/{job_id}/result
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
- `inout_judged.csv` when a court config is supplied
- `inout_overlay.avi` when a court config is supplied and `render_inout_video` is true
- `job.json`

### Auto-detect court config from a frame

Use this when the app has a still frame where the AprilTags are visible:

```text
POST /sessions/{session_id}/court-config/detect
multipart form:
  frame=<image file>
query params:
  family=tag36h11
  min_side_px=0
```

The backend saves the uploaded frame, detects AprilTag-guided view2 court lines, writes
`court_config_detected.json`, and stores that path on the session. Later
`POST /sessions/{session_id}/judge` automatically uses that detected config unless the
judge request overrides `court_config_path`.

## CLI: Y-Reversal Bounce Detection

```powershell
python -m src.bounce_detection.detect_bounces_from_track_csv --video_path ".\input\slow_inputs_seg\3대 작은마커\try5\try5_seg08_slow_0_5x.avi" --output_root .\output\output_y_reversal_bounces --device cuda
```
