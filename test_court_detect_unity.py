from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEPS = ROOT / ".codex_deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import cv2

from src.line_detection import detect_view2_apriltag_lines as view2
from src.line_detection import detect_view3_apriltag_line as view3
from src.line_detection import overlay_apriltag_lines as overlay


INPUT_ROOT = ROOT / "slow_inputs_unity"
OUT_ROOT = ROOT / "line_detection_output_unity_test"
FRAMES_DIR = OUT_ROOT / "frames"
JSON_DIR = OUT_ROOT / "json"
OVERLAY_DIR = OUT_ROOT / "overlays"
UNITY_TAG_FAMILY = "aruco_mip_36h12"
SAMPLE_COUNT = 9


def safe_stem(path: Path) -> str:
    relative = path.relative_to(INPUT_ROOT)
    return "__".join(part.replace(" ", "_") for part in relative.with_suffix("").parts)


def marker_count(frame) -> int:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(view2.family_dictionary(UNITY_TAG_FAMILY))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    _, ids, _ = detector.detectMarkers(gray)
    return 0 if ids is None else len(ids)


def extract_best_sample_frame(video_path: Path, frame_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if frame_count <= 1:
        sample_indices = [0]
    else:
        sample_indices = sorted(
            {
                round(i * (frame_count - 1) / max(SAMPLE_COUNT - 1, 1))
                for i in range(SAMPLE_COUNT)
            }
        )

    best_frame = None
    best_index = 0
    best_marker_count = -1
    samples = []
    for sample_index in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            samples.append({"frame": sample_index, "status": "read_error"})
            continue
        count = marker_count(frame)
        samples.append({"frame": sample_index, "marker_count": count})
        if count > best_marker_count:
            best_marker_count = count
            best_index = sample_index
            best_frame = frame
    cap.release()

    if best_frame is None:
        raise RuntimeError(f"Could not read frame from video: {video_path}")

    frame_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(frame_path.suffix, best_frame)
    if not ok:
        raise RuntimeError(f"Could not encode frame: {frame_path}")
    frame_path.write_bytes(encoded.tobytes())
    return {
        "frame_count": frame_count,
        "fps": fps,
        "sampled_frame": best_index,
        "sampled_marker_count": best_marker_count,
        "sample_candidates": samples,
        "frame_image": str(frame_path),
    }


def run_detector(mode: str, frame_path: Path, stem: str) -> dict:
    out_json = JSON_DIR / f"{stem}_{mode}.json"
    try:
        if mode == "view2":
            record = view2.process_image(frame_path, family=UNITY_TAG_FAMILY, min_side_px=0.0)
        elif mode == "view3":
            record = view3.process_image(
                frame_path,
                family=UNITY_TAG_FAMILY,
                min_side_px=0.0,
                line_marker_ids=None,
                vertical_weight=30.0,
            )
        else:
            raise ValueError(mode)

        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump([record], f, ensure_ascii=False, indent=2)

        overlay_dir = OVERLAY_DIR / mode
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_dir / f"{stem}_{mode}_overlay.png"
        image = overlay.read_image_exif(frame_path)
        overlay_image = overlay.draw_overlay(image, record, debug=True, thickness=6)
        ok, encoded = cv2.imencode(".png", overlay_image)
        if not ok:
            raise RuntimeError(f"Could not encode overlay: {overlay_path}")
        overlay_path.write_bytes(encoded.tobytes())
        return {
            "mode": mode,
            "status": "ok",
            "marker_count": record["marker_count"],
            "line_count": len(record.get("lines", [])),
            "json": str(out_json),
            "overlay": str(overlay_path),
        }
    except Exception as exc:
        return {
            "mode": mode,
            "status": "error",
            "error": str(exc),
        }


def main() -> None:
    videos = sorted(INPUT_ROOT.rglob("*.avi"))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    results = []
    for video_path in videos:
        stem = safe_stem(video_path)
        frame_path = FRAMES_DIR / f"{stem}_mid.png"
        result = {
            "video": str(video_path),
            "relative_video": str(video_path.relative_to(INPUT_ROOT)),
        }
        try:
            result.update(extract_best_sample_frame(video_path, frame_path))
            result["detections"] = [
                run_detector("view2", frame_path, stem),
                run_detector("view3", frame_path, stem),
            ]
        except Exception as exc:
            result["detections"] = []
            result["error"] = str(exc)
        results.append(result)

    summary_path = OUT_ROOT / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
