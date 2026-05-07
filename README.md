# Tennis Court Line Detection

## Run

```powershell
conda run -n courtcv python detect_baseline_singles_cv2.py --input "path\to\image.jpg" --out-dir "line_detection_output"
```

## Run With Roboflow ROI

```powershell
$env:ROBOFLOW_API_KEY="your_api_key"

conda run -n courtcv python detect_baseline_singles_cv2.py --input "path\to\image.jpg" --out-dir "line_detection_output" --use-roboflow-roi
```

## Save Debug Images

```powershell
conda run -n courtcv python detect_baseline_singles_cv2.py --input "path\to\image.jpg" --out-dir "line_detection_output" --use-roboflow-roi --save-debug
```

## Output

```text
near_three_lines.png
near_three_line_regions.png
detected_lines.json
```
