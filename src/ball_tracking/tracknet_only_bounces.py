import argparse
import csv
from collections import Counter
from pathlib import Path

import src.ball_tracking.infer_on_video as infer_module
from src.ball_tracking.infer_on_video import infer_model, read_video
from src.bounce_detection.detect_bounces import (
    add_contact_points,
    detect_bounces,
    load_model,
    scale_track_to_frame,
    select_device,
    write_bounce_csv,
    write_bounce_video,
)
from src.bounce_detection.detect_bounces_from_track_csv import write_track_csv


def iter_videos(input_root: Path, pattern: str) -> list[Path]:
    if input_root.is_file():
        return [input_root]
    return sorted(
        path
        for path in input_root.rglob(pattern)
        if "__MACOSX" not in path.parts and not path.name.startswith("._")
    )


def direct_statuses(track: list[tuple[float | None, float | None]]) -> list[str]:
    return [
        "detected" if x is not None and y is not None else "missing"
        for x, y in track
    ]


def output_dir_for_video(output_root: Path, input_root: Path, video_path: Path) -> Path:
    if input_root.is_file():
        return output_root / video_path.stem
    try:
        relative = video_path.relative_to(input_root).with_suffix("")
    except ValueError:
        relative = Path(video_path.stem)
    return output_root / relative


def process_video(video_path: Path, input_root: Path, output_root: Path, model, args) -> dict:
    frames, fps = read_video(str(video_path))
    if not frames:
        raise RuntimeError(f"video has no readable frames: {video_path}")

    raw_track, _, scores = infer_model(
        frames,
        model,
        return_scores=True,
        min_confidence=args.min_confidence,
        centroid_radius=args.centroid_radius,
        relative_threshold=args.relative_threshold,
        max_candidates=args.max_candidates,
        max_candidate_dist=args.max_candidate_dist,
        score_distance_tradeoff=args.score_distance_tradeoff,
        visual_refine=not args.no_visual_refine,
        visual_roi_radius=args.visual_roi_radius,
        fast_ball=not args.no_fast_ball,
        fast_speed_threshold=args.fast_speed_threshold,
        fast_min_confidence=args.fast_min_confidence,
    )
    statuses = direct_statuses(raw_track)
    frame_height, frame_width = frames[0].shape[:2]
    frame_track = scale_track_to_frame(raw_track, frame_width, frame_height)

    out_dir = output_dir_for_video(output_root, input_root, video_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    track_csv = out_dir / f"{video_path.stem}_tracknet_only_track.csv"
    bounces_csv = out_dir / f"{video_path.stem}_tracknet_only_bounces.csv"
    result_video = out_dir / f"{video_path.stem}_tracknet_only_bounces.avi"

    write_track_csv(track_csv, frame_track, statuses, scores, fps)
    bounces, _ = detect_bounces(frame_track, statuses, fps, args)
    bounces = add_contact_points(bounces, frame_track, statuses, frame_height, args)
    write_bounce_csv(bounces_csv, bounces)
    if args.render_video:
        write_bounce_video(
            frames,
            frame_track,
            statuses,
            bounces,
            result_video,
            fps,
            show_track=True,
            trace=args.trace,
            bounce_display_window=args.bounce_display_window,
        )

    selected = max(bounces, key=lambda row: float(row.get("score", 0.0)), default=None)
    counts = Counter(statuses)
    return {
        "video": str(video_path),
        "result": "bounce_detected" if selected else "no_bounce",
        "bounce_count": len(bounces),
        "primary_frame": "" if selected is None else int(selected["frame_index"]),
        "primary_time_sec": "" if selected is None else float(selected["time_sec"]),
        "primary_x": "" if selected is None else float(selected["contact_x"]),
        "primary_y": "" if selected is None else float(selected["contact_y"]),
        "primary_score": "" if selected is None else float(selected["score"]),
        "fps": fps,
        "frames": len(frames),
        "detected": counts.get("detected", 0),
        "missing": counts.get("missing", 0),
        "track_csv": str(track_csv),
        "bounces_csv": str(bounces_csv),
        "result_video": str(result_video) if args.render_video else "",
        "error": "",
    }


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "video",
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
        "track_csv",
        "bounces_csv",
        "result_video",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run direct TrackNet detections only, then detect bounces without Kalman or gap prediction."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--pattern", type=str, default="*.mp4")
    parser.add_argument("--output-root", type=Path, default=Path("output/tracknet_only_bounces"))
    parser.add_argument("--model-path", type=Path, default=Path("weights/tracknet_pretrained.pt"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no-render-video", dest="render_video", action="store_false")

    parser.add_argument("--min-confidence", type=float, default=82)
    parser.add_argument("--centroid-radius", type=int, default=9)
    parser.add_argument("--relative-threshold", type=float, default=0.55)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--max-candidate-dist", type=float, default=340)
    parser.add_argument("--score-distance-tradeoff", type=float, default=0.35)
    parser.add_argument("--no-visual-refine", action="store_true")
    parser.add_argument("--visual-roi-radius", type=int, default=36)
    parser.add_argument("--no-fast-ball", action="store_true")
    parser.add_argument("--fast-speed-threshold", type=float, default=50)
    parser.add_argument("--fast-min-confidence", type=float, default=55)

    parser.add_argument("--bounce-window", type=int, default=3)
    parser.add_argument("--local-window", type=int, default=4)
    parser.add_argument("--min-local-points", type=int, default=5)
    parser.add_argument("--min-down-speed", type=float, default=3.0)
    parser.add_argument("--min-up-speed", type=float, default=2.0)
    parser.add_argument("--local-y-tolerance", type=float, default=10.0)
    parser.add_argument("--max-x-velocity-change", type=float, default=85.0)
    parser.add_argument("--min-direct-detections", type=int, default=2)
    parser.add_argument("--min-bounce-score", type=float, default=9.0)
    parser.add_argument("--min-bounce-gap", type=int, default=10)
    parser.add_argument("--bounce-contact-offset", type=float, default=-1)
    parser.add_argument("--trajectory-window", type=int, default=8)
    parser.add_argument("--min-trajectory-points", type=int, default=2)
    parser.add_argument("--trajectory-search-radius", type=float, default=3.0)
    parser.add_argument("--max-trajectory-gap", type=float, default=130.0)
    parser.add_argument("--trajectory-blend", type=float, default=0.5)
    parser.add_argument("--trace", type=int, default=7)
    parser.add_argument("--bounce-display-window", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    infer_module.device = device
    model = load_model(args.model_path, device)
    videos = iter_videos(args.input_root, args.pattern)
    if not videos:
        raise RuntimeError(f"no videos found: {args.input_root} {args.pattern}")

    rows = []
    for index, video_path in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] {video_path}", flush=True)
        try:
            row = process_video(video_path, args.input_root, args.output_root, model, args)
            print(
                f"  -> {row['result']} count={row['bounce_count']} "
                f"detected={row['detected']} missing={row['missing']}",
                flush=True,
            )
        except Exception as exc:
            row = {
                "video": str(video_path),
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
                "track_csv": "",
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
