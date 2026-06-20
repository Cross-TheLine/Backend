import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.ball_tracking.yolo_tracker import (
    YOLO_TRACK_FIELDNAMES,
    detect_ball_yolo,
    detection_to_track_row,
)


def iter_videos(input_root: Path, pattern: str) -> list[Path]:
    if input_root.is_file():
        return [input_root]
    return sorted(
        path
        for path in input_root.rglob(pattern)
        if "__MACOSX" not in path.parts and not path.name.startswith("._")
    )


def output_dir_for_video(output_root: Path, input_root: Path, video_path: Path) -> Path:
    if input_root.is_file():
        return output_root / video_path.stem
    try:
        relative = video_path.relative_to(input_root).with_suffix("")
    except ValueError:
        relative = Path(video_path.stem)
    return output_root / relative


def draw_trail(frame, rows, current_index: int, trace: int) -> None:
    for offset in range(trace):
        index = current_index - offset
        if index < 0:
            break
        row = rows[index]
        if row["status"] != "detected":
            continue
        alpha = 1.0 - offset / max(trace, 1)
        color = (0, int(80 + 175 * alpha), 255)
        thickness = max(2, int(8 - offset * 0.8))
        cv2.circle(frame, (int(row["x"]), int(row["y"])), 4, color, thickness)


def process_video(video_path: Path, input_root: Path, output_root: Path, model: YOLO, args) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dir = output_dir_for_video(output_root, input_root, video_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    track_csv = out_dir / f"{video_path.stem}_yolo_track.csv"
    result_video = out_dir / f"{video_path.stem}_yolo_track.avi"

    writer = None
    if args.render_video:
        writer = cv2.VideoWriter(
            str(result_video),
            cv2.VideoWriter_fourcc(*"DIVX"),
            fps,
            (width, height),
        )

    rows = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detection = detect_ball_yolo(
            frame,
            model,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            max_det=args.max_det,
            class_id=args.class_id,
        )
        row = detection_to_track_row(frame_index, fps, detection)
        rows.append(row)

        if writer is not None:
            canvas = frame.copy()
            draw_trail(canvas, rows, frame_index, args.trace)
            if detection is not None:
                x1, y1, x2, y2 = (
                    int(detection["x1"]),
                    int(detection["y1"]),
                    int(detection["x2"]),
                    int(detection["y2"]),
                )
                x, y = int(detection["x"]), int(detection["y"])
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(
                    canvas,
                    f"YOLO ball {detection['confidence']:.2f}",
                    (max(0, x1), max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(canvas)
        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()

    with track_csv.open("w", newline="", encoding="utf-8-sig") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=YOLO_TRACK_FIELDNAMES)
        csv_writer.writeheader()
        for row in rows:
            csv_writer.writerow({
                key: round(float(row[key]), 4)
                if key not in {"frame_index", "status", "class_id"} and row[key] != ""
                else row[key]
                for key in YOLO_TRACK_FIELDNAMES
            })

    detected = sum(1 for row in rows if row["status"] == "detected")
    best = max(
        (row for row in rows if row["status"] == "detected"),
        key=lambda row: float(row["confidence"]),
        default=None,
    )
    return {
        "video": str(video_path),
        "result": "detected" if detected else "no_detection",
        "frames": frame_index or frame_count,
        "detected": detected,
        "missing": max((frame_index or frame_count) - detected, 0),
        "best_frame": "" if best is None else int(best["frame_index"]),
        "best_time_sec": "" if best is None else float(best["time_sec"]),
        "best_x": "" if best is None else float(best["x"]),
        "best_y": "" if best is None else float(best["y"]),
        "best_confidence": "" if best is None else float(best["confidence"]),
        "track_csv": str(track_csv),
        "result_video": str(result_video) if args.render_video else "",
        "error": "",
    }


def write_summary(path: Path, rows: list[dict]) -> None:
    fields = [
        "video",
        "result",
        "frames",
        "detected",
        "missing",
        "best_frame",
        "best_time_sec",
        "best_x",
        "best_y",
        "best_confidence",
        "track_csv",
        "result_video",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track tennis balls with a YOLO detector.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--pattern", type=str, default="*.mp4")
    parser.add_argument("--output-root", type=Path, default=Path("output/yolo_ball_tracking"))
    parser.add_argument("--model-path", type=Path, default=Path("weights/tennis_ball_yolo.pt"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--max-det", type=int, default=10)
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--trace", type=int, default=8)
    parser.add_argument("--no-render-video", dest="render_video", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model_path))
    print(f"model_names={model.names}", flush=True)

    videos = iter_videos(args.input_root, args.pattern)
    if not videos:
        raise RuntimeError(f"no videos found: {args.input_root} {args.pattern}")

    rows = []
    for index, video_path in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] {video_path}", flush=True)
        try:
            row = process_video(video_path, args.input_root, args.output_root, model, args)
            print(
                f"  -> {row['result']} detected={row['detected']}/{row['frames']} "
                f"best_conf={row['best_confidence']}",
                flush=True,
            )
        except Exception as exc:
            row = {
                "video": str(video_path),
                "result": "error",
                "frames": "",
                "detected": "",
                "missing": "",
                "best_frame": "",
                "best_time_sec": "",
                "best_x": "",
                "best_y": "",
                "best_confidence": "",
                "track_csv": "",
                "result_video": "",
                "error": repr(exc),
            }
            print(f"  -> error {exc!r}", flush=True)
        rows.append(row)

    summary_path = args.output_root / "summary.csv"
    write_summary(summary_path, rows)
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
