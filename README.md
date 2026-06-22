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

AprilTag 라인 인식 결과와 YOLO 기반 공 궤적/bounce 결과를 같은 픽셀 좌표계에서 묶어, 바운스 지점이 라인 안쪽인지 바깥쪽인지 판정할 수 있습니다.

## Setup

Python 3.11+ 권장.

```powershell
pip install -r requirements.txt
```

로컬에 이미 `tennis-env` conda 환경이 있으면 아래처럼 실행해도 됩니다.

```powershell
conda run -n tennis-env python -m <module> ...
```

NVIDIA GPU를 사용한다면 OS, Python 버전, GPU 드라이버 및 지원 CUDA 버전에 맞는
PyTorch wheel을 설치하세요. 예를 들어 CUDA 12.8 환경에서는 다음 명령을 사용할 수 있습니다.

```powershell
pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

CUDA wheel을 선택하기 전에 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)를 확인하세요.

필요한 모델 파일:

```text
weights/tennis_ball_yolo.pt
```

## Folders

- `input/slow_inputs_seg/`: segmented input videos (필요할 때 로컬에서 생성).
- `input/slow_inputs_unity/`: Unity input videos (필요할 때 로컬에서 생성).
- `output/`: generated result folders and archives (실행 중 자동 생성).
- `src/ball_tracking/`: YOLO ball tracking core and video helpers.
- `src/bounce_detection/`: y-reversal bounce detection from tracked coordinates.
- `src/inout_judgement/`: in/out judgment and combined overlay helpers.
- `src/line_detection/`: AprilTag and court-line detection helpers.
- `src/api/`: FastAPI server.

`input/`, `output/`, `weights/` 디렉터리는 `.gitignore` 대상이므로 새로 clone한 저장소에는 포함되지 않습니다.

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

`judge_ready` is true only when `torch`, `ultralytics`, and `weights/tennis_ball_yolo.pt` are available.

## MVP App Flow

백엔드가 로컬 파일시스템 경로를 통해 전체 녹화 영상에 접근할 수 있고, 앱에서 버튼 입력 시각을 전달하는 경우 사용합니다.

```text
POST /judge-preprocess
GET  /jobs/{job_id}
GET  /jobs/{job_id}/result
```

Request:

`recording_path`와 `court_config_path`는 모두 백엔드 서버의 로컬 경로입니다.
클라이언트 기기의 경로가 아니며, 이 엔드포인트는 파일을 업로드하지 않습니다.

```json
{
  "recording_path": "C:/path/to/full_recording.mp4",
  "use_video_end": true,
  "lookback_sec": 5.0,
  "render_video": true,
  "court_config_path": "C:/path/to/view2_lines.json",
  "config_image": null,
  "config_index": 0,
  "render_inout_video": true
}
```

앱이 판정 버튼을 누를 때 녹화를 중지한다면 `pressed_at_sec`을 생략하거나
`use_video_end=true`로 설정합니다. 백엔드는 영상의 `video_duration_sec`을 읽고
마지막 `lookback_sec` 구간을 판정합니다.

```text
pressed_at_sec = video_duration_sec - end_offset_sec
clip_start_sec = max(0, pressed_at_sec - lookback_sec)
```

기본 `lookback_sec=5.0`에서 5초보다 짧은 영상은 영상 시작부터 판정합니다.

앱이 버튼 입력 이후에도 잠시 녹화하는 경우에만 `end_offset_sec`을 사용합니다.
예를 들어 입력 후 0.5초 뒤 녹화가 종료된다면 `"end_offset_sec": 0.5`를 전달합니다.

직접 시간을 동기화하는 클라이언트는 `pressed_at_sec`을 사용할 수 있습니다.
영상 길이를 초과하면 백엔드가 영상 끝 시각으로 제한합니다.

Initial response:

```json
{
  "job_id": "uuid",
  "status": "pending"
}
```

`GET /jobs/{job_id}`를 `status`가 `done` 또는 `failed`가 될 때까지 polling하거나,
프론트엔드용 고정 응답 구조가 필요하면 `GET /jobs/{job_id}/result`를 사용합니다.

Frontend result response:

```json
{
  "job_id": "uuid",
  "session_id": "uuid",
  "status": "done",
  "result": "bounce_detected",
  "decision": "IN",
  "is_in": true,
  "failure_type": null,
  "failure_reason": null,
  "video_readable": true,
  "recording_accessible": true,
  "confidence": 0.328,
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

`POST /judge-preprocess`에서 `court_config_path`를 생략하면 IN/OUT 판정 없이 공 추적과
바운스 검출만 실행합니다. 세션 기반 흐름에서는 경로를 생략할 경우 해당 세션에 저장된
court config를 사용합니다.

## Session-Based Flow

백엔드가 세션 기록을 관리해야 할 때 사용합니다.

```text
POST /sessions/start
POST /sessions/{session_id}/court-config/detect
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

모든 job은 `job.json`을 생성합니다. 처리가 성공한 job은 다음 파일도 생성합니다.

- `clip.mp4`
- `track.csv`
- `bounces.csv`
- `result.avi` when `render_video` is true
- `inout_judged.csv` when a court config is supplied
- `inout_overlay.avi` when a court config is supplied and `render_inout_video` is true

실패한 job은 실패 지점에 따라 `job.json`만 생성되거나 일부 산출물만 생성될 수 있습니다.

### Auto-detect court config from a frame

지원되는 view2 마커 배치가 포함된 정지 프레임이 있을 때 사용합니다. 감지에는 최소 3개의
AprilTag가 보여야 하며, 감지된 태그 중 가장 큰 3개를 기준으로 두 코트 경계선을 추정합니다.
일반적인 코트 라인 검출 기능은 아닙니다.

```text
POST /sessions/{session_id}/court-config/detect
multipart form:
  frame=<image file>
query params:
  family=tag36h11
  min_side_px=0
```

백엔드는 업로드된 프레임을 저장하고 AprilTag 기반 view2 코트 라인을 감지한 다음
`court_config_detected.json`을 생성하여 세션에 경로를 저장합니다.
이후 `POST /sessions/{session_id}/judge`는 요청에서 `court_config_path`를 재정의하지 않는 한
세션에 저장된 설정을 자동으로 사용합니다.

### Save judgment records

job이 `status=done`에 도달한 뒤 판정 결과를 기록에 저장할 때 사용합니다.

```text
POST /sessions/{session_id}/save
```

Request:

```json
{
  "match_type": "singles",
  "recorded_at": "2026-06-07T12:30:00+09:00",
  "recorded_date": "2026-06-07"
}
```

`match_type`은 `singles` 또는 `doubles`여야 합니다. `recorded_at`을 생략하면 백엔드는
세션의 `recorded_at`을 먼저 사용하고, 값이 없으면 저장 시각을 사용합니다.

The saved SQLite record includes:

- original recording path and, for videos received through `record/upload`, a `/files` URL
- judgment clip and overlay URLs
- match type: singles or doubles
- primary IN/OUT decision and reason
- primary bounce and full job result JSON

`POST /sessions/{session_id}/save`는 세션에서 최근 완료된 job을 저장합니다.
완료된 특정 job을 저장하려면 선택적 `job_id`를 전달할 수 있습니다.

Records are stored in:

```text
output/judgements.sqlite3
```

## Notes

- 현재 bounce detection은 homography를 쓰지 않고, 원본 영상 픽셀 좌표에서 y-reversal을 봅니다.
- in/out도 같은 픽셀 좌표계에서 line JSON polygon 안/밖을 봅니다.
- 정확한 실제 코트 좌표, 거리, cm 단위 margin이 필요하면 별도 homography 단계가 필요할 수 있습니다.
- 라인 근처 판정은 ball center/contact point 오차 영향을 받을 수 있습니다.


## 기여자 및 TrackNet 관련 안내

이 저장소는 초기 개발 과정에서 TrackNet 저장소를 기반으로 구성되었기 때문에 GitHub 기여자 기록에 TrackNet 원본 기여자가 표시될 수 있습니다.
현재 구현에서는 TrackNet 모델과 추론 파이프라인을 더 이상 사용하지 않습니다. 공 추적은 `src/ball_tracking/`의 YOLO 기반 구현을 사용합니다.
