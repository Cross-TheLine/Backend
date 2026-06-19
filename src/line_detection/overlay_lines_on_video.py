from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_LINE_THICKNESS = 3


def rounded_int_point(point: list[float] | tuple[float, float]) -> tuple[int, int]:
    return tuple(np.round(np.array(point, dtype=np.float32)).astype(int))


def load_lines(line_json: Path) -> list[dict]:
    with line_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    records = data if isinstance(data, list) else [data]
    lines: list[dict] = []
    for record in records:
        lines.extend(record.get("lines", []))
    if not lines:
        raise ValueError(f"No lines found in {line_json}")
    return lines


def draw_lines(frame: np.ndarray, lines: list[dict], color: tuple[int, int, int], thickness: int) -> np.ndarray:
    output = frame.copy()
    for line in lines:
        endpoints = line.get("extended_endpoints") or line.get("segment_endpoints")
        if not endpoints or len(endpoints) != 2:
            continue
        cv2.line(
            output,
            rounded_int_point(endpoints[0]),
            rounded_int_point(endpoints[1]),
            color,
            thickness,
            cv2.LINE_AA,
        )
    return output


def process_video(video_path: Path, line_json: Path, out_path: Path, color: tuple[int, int, int], thickness: int) -> dict:
    lines = load_lines(line_json)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video writer: {out_path}")

    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(draw_lines(frame, lines, color, thickness))
        written += 1

    cap.release()
    writer.release()

    return {
        "input": str(video_path),
        "output": str(out_path),
        "line_json": str(line_json),
        "width": width,
        "height": height,
        "fps": fps,
        "input_frame_count": frame_count,
        "written_frame_count": written,
        "line_count": len(lines),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw line-detection JSON lines onto video frames.")
    parser.add_argument("--line-json", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--line-thickness", type=int, default=DEFAULT_LINE_THICKNESS)
    parser.add_argument("--color-bgr", default="255,0,0", help="B,G,R line color.")
    args = parser.parse_args()

    color_parts = [int(part.strip()) for part in args.color_bgr.split(",")]
    if len(color_parts) != 3:
        raise ValueError("--color-bgr must have exactly three comma-separated integers")
    color = tuple(color_parts)

    out_dir = Path(args.out_dir)
    results = []
    for input_path in [Path(path) for path in args.inputs]:
        out_path = out_dir / f"{input_path.stem}_lines.mp4"
        results.append(
            process_video(
                video_path=input_path,
                line_json=Path(args.line_json),
                out_path=out_path,
                color=color,
                thickness=args.line_thickness,
            )
        )

    summary_path = out_dir / "line_video_overlay_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
