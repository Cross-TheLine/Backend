# Cross The Line Backend

Tennis ball tracking, bounce detection, line detection, and in/out judgment for segmented tennis videos.

## Environment

- Python 3.11+
- Install packages:

```powershell
pip install -r requirements.txt
```

For NVIDIA GPU inference, install the CUDA PyTorch wheel:

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
- `src/inout_judgement/`: in/out judgment from bounce coordinates.
- `src/line_detection/`: AprilTag and court-line detection helpers.

## Core Files

- `src/ball_tracking/infer_on_video.py`: TrackNet ball tracking and trajectory post-processing.
- `src/ball_tracking/model.py`: TrackNet model definition.
- `src/ball_tracking/general.py`: TrackNet heatmap post-processing helpers.
- `src/bounce_detection/detect_bounces.py`: Existing y-velocity reversal bounce detector.
- `src/bounce_detection/detect_bounces_from_track_csv.py`: Saves TrackNet coordinates to CSV, then detects bounces from y-value reversal.
- `src/inout_judgement/judge_in_out.py`: Optional in/out judgment from bounce CSVs.
- `src/line_detection/*.py`: AprilTag and court-line detection utilities.

## Run Ball Tracking

```powershell
python -m src.ball_tracking.infer_on_video --model_path .\weights\tracknet_pretrained.pt --video_path .\input\slow_inputs_seg\video.avi --video_out_path .\output\video_track.avi --device cuda
```

## Run Y-Reversal Bounce Detection

This uses TrackNet coordinates and detects the point where the ball falls downward
then rebounds upward:

```powershell
python -m src.bounce_detection.detect_bounces_from_track_csv --video_path ".\input\slow_inputs_seg\3대 작은마커\try5\try5_seg08_slow_0_5x.avi" --output_root .\output\output_y_reversal_bounces --device cuda
```

Outputs:

- `<video>_track.csv`
- `<video>_y_bounces.csv`
- `<video>_y_bounces.avi`
