# Cross The Line Backend

테니스 영상에서 AprilTag 기반 라인을 잡고, TrackNet으로 공 궤적과 바운스를 찾은 뒤,
바운스 지점이 라인 안쪽인지 바깥쪽인지 판정하는 백엔드 코드입니다.

## Setup

Python 3.11+ 권장.

```powershell
pip install -r requirements.txt
```

GPU로 돌릴 경우 CUDA PyTorch wheel을 따로 설치합니다.

```powershell
pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

필요한 모델 파일:

```text
weights/tracknet_pretrained.pt
```

로컬에 이미 `tennis-env` conda 환경이 있으면 아래처럼 실행해도 됩니다.

```powershell
conda run -n tennis-env python -m <module> ...
```

## Pipeline

전체 흐름은 아래 순서입니다.

```text
video/image
-> line detection input frame
-> view2 line JSON
-> bounce detection CSV
-> in/out judgment CSV
-> combined overlay video
```

영상 자르기는 기본 line/bounce/inout 코드 안에 넣지 않았습니다. 영상이면 먼저 한 프레임을 이미지로 뽑고, 사진이면 그 사진을 바로 line detection에 넣으면 됩니다.

## Main Files

- `src/inout_judgement/extract_video_frame.py`: 영상에서 라인 인식용 프레임 1장 추출
- `src/line_detection/detect_view2_apriltag_lines.py`: view2 AprilTag 라인 JSON 생성
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

출력 JSON 예:

```text
outputs/xyline_yball_view2_lines.json
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

GPU 환경이면 `--device cuda`를 사용합니다.

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
