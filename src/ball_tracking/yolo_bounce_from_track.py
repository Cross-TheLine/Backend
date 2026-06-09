import argparse
import csv
from pathlib import Path

from src.ball_tracking.infer_on_video import read_video
from src.bounce_detection.detect_bounces_from_track_csv import (
    detect_y_reversal_bounces,
    write_bounce_csv,
    write_video,
)


def load_yolo_track(path: Path) -> tuple[list[tuple[float | None, float | None]], list[str], list[float]]:
    rows = []
    statuses = []
    scores = []
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        for row in csv.DictReader(csv_file):
            status = row.get("status") or "missing"
            if status == "detected" and row.get("x") and row.get("y"):
                rows.append((float(row["x"]), float(row["y"])))
                scores.append(float(row.get("confidence") or 0.0))
                statuses.append("detected")
            else:
                rows.append((None, None))
                scores.append(0.0)
                statuses.append("missing")
    return rows, statuses, scores


def find_video_for_track(track_csv: Path, input_root: Path) -> Path:
    stem = track_csv.name.removesuffix("_yolo_track.csv")
    candidates = sorted(input_root.rglob(f"{stem}.mp4"))
    if not candidates:
        candidates = sorted(input_root.rglob(f"{stem}.*"))
    if not candidates:
        raise FileNotFoundError(f"could not find source video for {track_csv}")
    return candidates[0]


def output_dir_for_track(output_root: Path, track_root: Path, track_csv: Path) -> Path:
    try:
        relative = track_csv.parent.relative_to(track_root)
    except ValueError:
        relative = Path(track_csv.stem)
    return output_root / relative


def process_track(track_csv: Path, track_root: Path, input_root: Path, output_root: Path, args) -> dict:
    video_path = find_video_for_track(track_csv, input_root)
    frames, fps = read_video(str(video_path))
    if not frames:
        raise RuntimeError(f"video has no readable frames: {video_path}")

    track, statuses, scores = load_yolo_track(track_csv)
    out_dir = output_dir_for_track(output_root, track_root, track_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = track_csv.name.removesuffix("_yolo_track.csv")
    bounce_csv = out_dir / f"{stem}_yolo_bounces.csv"
    result_video = out_dir / f"{stem}_yolo_bounces.avi"

    bounces = detect_y_reversal_bounces(track, fps, args)
    write_bounce_csv(bounce_csv, bounces)
    if args.render_video:
        write_video(frames, track, bounces, result_video, fps)

    selected = max(bounces, key=lambda row: float(row.get("score", 0.0)), default=None)
    detected = statuses.count("detected")
    return {
        "video": str(video_path),
        "track_csv": str(track_csv),
        "result": "bounce_detected" if selected else "no_bounce",
        "bounce_count": len(bounces),
        "primary_frame": "" if selected is None else int(selected["frame_index"]),
        "primary_time_sec": "" if selected is None else float(selected["time_sec"]),
        "primary_x": "" if selected is None else float(selected["x"]),
        "primary_y": "" if selected is None else float(selected["y"]),
        "primary_score": "" if selected is None else float(selected["score"]),
        "fps": fps,
        "frames": len(frames),
        "detected": detected,
        "missing": max(len(track) - detected, 0),
        "bounces_csv": str(bounce_csv),
        "result_video": str(result_video) if args.render_video else "",
        "error": "",
    }


def write_summary(path: Path, rows: list[dict]) -> None:
    fields = [
        "video",
        "track_csv",
        "result",
        "bounce_count",
        "primary_frame",
        "primary_time_sec",
        "primary_x",
        "primary_y",
        "primary_score",
        "fps",
        "frames",
        "detected",
        "missing",
        "bounces_csv",
        "result_video",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect bounces from YOLO ball track CSV files.")
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("output/yolo_ball_bounces"))
    parser.add_argument("--pattern", type=str, default="*_yolo_track.csv")
    parser.add_argument("--no-render-video", dest="render_video", action="store_false")

    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--local-window", type=int, default=4)
    parser.add_argument("--min-local-points", type=int, default=5)
    parser.add_argument("--min-down-speed", type=float, default=3.0)
    parser.add_argument("--min-up-speed", type=float, default=2.0)
    parser.add_argument("--local-y-tolerance", type=float, default=10.0)
    parser.add_argument("--max-x-velocity-change", type=float, default=85.0)
    parser.add_argument("--min-score", type=float, default=9.0)
    parser.add_argument("--min-gap", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracks = sorted(args.track_root.rglob(args.pattern))
    if not tracks:
        raise RuntimeError(f"no YOLO track CSV files found: {args.track_root}")

    rows = []
    for index, track_csv in enumerate(tracks, start=1):
        print(f"[{index}/{len(tracks)}] {track_csv}", flush=True)
        try:
            row = process_track(track_csv, args.track_root, args.input_root, args.output_root, args)
            print(f"  -> {row['result']} count={row['bounce_count']}", flush=True)
        except Exception as exc:
            row = {
                "video": "",
                "track_csv": str(track_csv),
                "result": "error",
                "bounce_count": "",
                "primary_frame": "",
                "primary_time_sec": "",
                "primary_x": "",
                "primary_y": "",
                "primary_score": "",
                "fps": "",
                "frames": "",
                "detected": "",
                "missing": "",
                "bounces_csv": "",
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
