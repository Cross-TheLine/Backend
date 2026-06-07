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

AprilTag 라인 인식 결과와 TrackNet 기반 공 궤적/bounce 결과를 같은 픽셀 좌표계에서 묶어, 바운스 지점이 라인 안쪽인지 바깥쪽인지 판정할 수 있습니다.

## Setup

Python 3.11+ 권장.

```powershell
pip install -r requirements.txt
```

로컬에 이미 `tennis-env` conda 환경이 있으면 아래처럼 실행해도 됩니다.

```powershell
conda run -n tennis-env python -m <module> ...
```

If you use an NVIDIA GPU, replace the default PyTorch install with the CUDA wheel:

```powershell
pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

필요한 모델 파일:

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

## CLI Pipeline

전체 CLI 흐름은 아래 순서입니다.

```text
video/image
-> line detection input frame
-> view2 or view3 line JSON
-> bounce detection CSV
-> in/out judgment CSV
-> combined overlay video
```

영상 자르기는 기본 line/bounce/inout 코드 안에 넣지 않았습니다. 영상이면 먼저 한 프레임을 이미지로 뽑고, 사진이면 그 사진을 바로 line detection에 넣으면 됩니다.

## Main CLI Files

- `src/inout_judgement/extract_video_frame.py`: 영상에서 라인 인식용 프레임 1장 추출
- `src/line_detection/detect_view2_apriltag_lines.py`: view2 AprilTag 라인 JSON 생성
- `src/line_detection/detect_view3_apriltag_line.py`: view3 AprilTag 라인 JSON 생성
- `src/line_detection/overlay_apriltag_lines.py`: 라인 인식 결과 이미지 overlay 생성
- `src/bounce_detection/detect_bounces_from_track_csv.py`: TrackNet 좌표 추정 + y-reversal bounce detect
- `src/inout_judgement/judge_in_out.py`: bounce CSV와 line JSON으로 `IN` / `OUT` 판정
- `src/inout_judgement/overlay_in_out.py`: 라인, 공 궤적, bounce, in/out 결과를 영상에 합쳐서 그림

## 1. Extract A Frame From Video

영상 입력이면 초반 프레임을 이미지로 추출합니다.

```powershell
python -m src.inout_judgement.extract_video_frame --videos .\xyLine\yball\mid.mov --out-dir .\outputs\xyline_yball_frames --frame-index 0
```

출력 예:

```text
outputs/xyline_yball_frames/mid_frame0.png
```

이미 사진 파일이 있으면 이 단계는 건너뛰고, 그 사진을 다음 단계 `--inputs`에 넣으면 됩니다.

## 2. Detect View2 Lines

추출한 프레임이나 사진에서 AprilTag 3개를 찾아 view2 라인 JSON을 만듭니다.

```powershell
python -m src.line_detection.detect_view2_apriltag_lines --inputs .\outputs\xyline_yball_frames\mid_frame0.png --out-json .\outputs\xyline_yball_view2_lines.json
```

라인이 잘 잡혔는지 확인하고 싶으면 overlay 이미지를 만듭니다.

```powershell
python -m src.line_detection.overlay_apriltag_lines --line-json .\outputs\xyline_yball_view2_lines.json --out-dir .\outputs\xyline_yball_view2_overlay --line-thickness 6
```

## 3. Detect Bounces

영상에서 공을 추적하고 y축 방향 반전으로 bounce를 찾습니다.

```powershell
python -m src.bounce_detection.detect_bounces_from_track_csv --video_path .\xyLine\yball\mid.mov --output_root .\outputs\xyline_yball_bounces --device cpu
```

GPU 환경이면 CUDA PyTorch 설치 후 `--device cuda`를 사용합니다.

출력:

```text
outputs/xyline_yball_bounces/mid_track.csv
outputs/xyline_yball_bounces/mid_y_bounces.csv
outputs/xyline_yball_bounces/mid_y_bounces.avi
```

`mid_y_bounces.csv`의 `x`, `y`는 bounce 판정에 사용할 픽셀 좌표입니다.

## 4. Judge In/Out

view2 line JSON으로 코트 안쪽 polygon을 만들고, bounce 좌표가 그 안에 있으면 `IN`, 밖이면 `OUT`으로 판정합니다.
view3 line JSON은 한 개의 끝-끝 라인과 `view_side`/`inside_point`를 사용해 안쪽 half-plane polygon을 만든 뒤 같은 방식으로 판정합니다.

```powershell
python -m src.inout_judgement.judge_in_out --input .\outputs\xyline_yball_bounces\mid_y_bounces.csv --court_config .\outputs\xyline_yball_view2_lines.json --output_root .\outputs\xyline_yball_inout --x_column x --y_column y --config_image mid_frame0.png
```

출력:

```text
outputs/xyline_yball_inout/mid_y_bounces_judged.csv
```

중요 컬럼:

- `decision`: `IN`, `OUT`, `UNKNOWN`
- `decision_reason`: `inside_polygon`, `outside_polygon`, `on_line`
- `boundary_distance_px`: 라인/polygon 경계와의 픽셀 거리

참고: `detect_bounces_from_track_csv` 결과는 좌표 컬럼이 `x/y`라서 `--x_column x --y_column y`가 필요합니다. `contact_x/contact_y`가 있는 CSV는 이 옵션을 빼도 됩니다.

## 5. Draw Combined Overlay

라인, 공 track, bounce 위치, `IN/OUT` 라벨을 원본 영상 위에 같이 그립니다.

```powershell
python -m src.inout_judgement.overlay_in_out --video_path .\xyLine\yball\mid.mov --court_config .\outputs\xyline_yball_view2_lines.json --judged_csv .\outputs\xyline_yball_inout\mid_y_bounces_judged.csv --track_csv .\outputs\xyline_yball_bounces\mid_track.csv --output_path .\outputs\xyline_yball_combined_overlay\mid_combined_inout.avi --config_image mid_frame0.png
```

출력:

```text
outputs/xyline_yball_combined_overlay/mid_combined_inout.avi
```

bounce가 없으면 영상에는 `NO BOUNCE`로 표시됩니다.

## Quick Example: xball in.mov

```powershell
python -m src.inout_judgement.extract_video_frame --videos .\xyLine\xball\in.mov --out-dir .\outputs\xyline_xball_frames --frame-index 0
python -m src.line_detection.detect_view2_apriltag_lines --inputs .\outputs\xyline_xball_frames\in_frame0.png --out-json .\outputs\xyline_xball_view2_lines.json
python -m src.bounce_detection.detect_bounces_from_track_csv --video_path .\xyLine\xball\in.mov --output_root .\outputs\xyline_xball_bounces --device cpu
python -m src.inout_judgement.judge_in_out --input .\outputs\xyline_xball_bounces\in_y_bounces.csv --court_config .\outputs\xyline_xball_view2_lines.json --output_root .\outputs\xyline_xball_inout --x_column x --y_column y --config_image in_frame0.png
python -m src.inout_judgement.overlay_in_out --video_path .\xyLine\xball\in.mov --court_config .\outputs\xyline_xball_view2_lines.json --judged_csv .\outputs\xyline_xball_inout\in_y_bounces_judged.csv --track_csv .\outputs\xyline_xball_bounces\in_track.csv --output_path .\outputs\xyline_xball_combined_overlay\in_combined_inout.avi --config_image in_frame0.png
```

## Notes

- 현재 bounce detection은 homography를 쓰지 않고, 원본 영상 픽셀 좌표에서 y-reversal을 봅니다.
- in/out도 같은 픽셀 좌표계에서 line JSON polygon 안/밖을 봅니다.
- 정확한 실제 코트 좌표, 거리, cm 단위 margin이 필요하면 별도 homography 단계가 필요할 수 있습니다.
- 라인 근처 판정은 ball center/contact point 오차 영향을 받을 수 있습니다.
