from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


CORNER_COLORS = [
    (0, 0, 255),
    (0, 180, 0),
    (255, 0, 0),
    (0, 220, 255),
]

DEFAULT_LINE_THICKNESS = 3


def read_image_exif(path: Path) -> np.ndarray:
    try:
        from PIL import Image, ImageOps

        pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            raise FileNotFoundError(path)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        return image


def rounded_int_point(point: list[float] | np.ndarray) -> tuple[int, int]:
    return tuple(np.round(np.array(point, dtype=np.float32)).astype(int))


def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return [data]


def marker_lookup(record: dict) -> dict[int, dict]:
    return {int(marker["marker_index"]): marker for marker in record.get("markers", [])}


def draw_debug_markers(image: np.ndarray, record: dict) -> None:
    role_indices = {
        int(role["marker_index"])
        for role in record.get("roles", {}).values()
        if isinstance(role, dict) and "marker_index" in role
    }
    for marker in record.get("markers", []):
        marker_index = int(marker["marker_index"])
        marker_id = marker["id"]
        corners = np.array(marker["corners"], dtype=np.float32)
        center = np.array(marker["center"], dtype=np.float32)
        outline_color = (0, 255, 255) if marker_index in role_indices else (255, 255, 0)
        cv2.polylines(image, [np.round(corners).astype(np.int32)], True, outline_color, 5, cv2.LINE_AA)
        cv2.putText(
            image,
            f"M{marker_index} ID {marker_id}",
            tuple(np.round(center + np.array([14, -16])).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
        for corner_idx, point in enumerate(corners):
            color = CORNER_COLORS[corner_idx]
            cv2.circle(image, rounded_int_point(point), 7, color, -1, cv2.LINE_AA)
            cv2.putText(
                image,
                f"c{corner_idx}",
                tuple(np.round(point + np.array([9, -9])).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )


def draw_debug_edges_and_points(image: np.ndarray, record: dict) -> None:
    markers = marker_lookup(record)
    for line in record.get("lines", []):
        for edge_info in line.get("edges", []):
            marker = markers.get(int(edge_info["marker_index"]))
            if marker is None:
                continue
            corners = np.array(marker["corners"], dtype=np.float32)
            edge = edge_info["edge"]
            points = corners[[int(edge[0]), int(edge[1])]]
            cv2.line(image, rounded_int_point(points[0]), rounded_int_point(points[1]), (0, 255, 255), 8, cv2.LINE_AA)

        segment = line.get("segment_endpoints")
        if segment is not None:
            cv2.line(image, rounded_int_point(segment[0]), rounded_int_point(segment[1]), (0, 255, 255), 5, cv2.LINE_AA)

        for point in line.get("fit_points", []):
            cv2.circle(image, rounded_int_point(point), 10, (0, 0, 255), -1, cv2.LINE_AA)


def draw_lines(image: np.ndarray, record: dict, thickness: int) -> None:
    for line in record.get("lines", []):
        endpoints = line.get("points") or line.get("extended_endpoints") or line.get("segment_endpoints")
        if endpoints is None:
            continue
        cv2.line(image, rounded_int_point(endpoints[0]), rounded_int_point(endpoints[1]), (255, 0, 0), thickness, cv2.LINE_AA)


def draw_overlay(image: np.ndarray, record: dict, debug: bool, thickness: int) -> np.ndarray:
    overlay = image.copy()
    if debug:
        draw_debug_markers(overlay, record)
        draw_debug_edges_and_points(overlay, record)
    draw_lines(overlay, record, thickness)
    return overlay


def image_path_for_record(record: dict, line_json: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    image_path = Path(record["image"])
    if image_path.is_absolute() or image_path.exists():
        return image_path
    candidate = line_json.parent / image_path
    if candidate.exists():
        return candidate
    return image_path


def output_path_for_image(out_dir: Path, image_path: Path, used_names: set[str]) -> Path:
    stem = image_path.stem.replace(" ", "_")
    name = f"{stem}_apriltag_lines_overlay.png"
    if name not in used_names:
        used_names.add(name)
        return out_dir / name

    suffix = 2
    while True:
        name = f"{stem}_{suffix}_apriltag_lines_overlay.png"
        if name not in used_names:
            used_names.add(name)
            return out_dir / name
        suffix += 1


def process_record(
    record: dict,
    line_json: Path,
    image_path: Path,
    out_dir: Path,
    used_names: set[str],
    debug: bool,
    thickness: int,
) -> dict:
    image = read_image_exif(image_path)
    overlay = draw_overlay(image, record, debug, thickness)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_path_for_image(out_dir, image_path, used_names)
    cv2.imwrite(str(overlay_path), overlay)
    return {
        "image": str(image_path),
        "schema": record.get("schema"),
        "mode": record.get("mode", record.get("view")),
        "method": record.get("method"),
        "view": record.get("view", record.get("mode")),
        "view_side": record.get("view_side"),
        "line_count": len(record.get("lines", [])),
        "line_json": str(line_json),
        "overlay": str(overlay_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw AprilTag line overlays from an existing JSON file.")
    parser.add_argument("--line-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="Optional replacement image paths. Count must match records in --line-json.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Also draw marker outlines, corner labels, selected edges, and fit points.",
    )
    parser.add_argument("--line-thickness", type=int, default=DEFAULT_LINE_THICKNESS)
    args = parser.parse_args()

    line_json = Path(args.line_json)
    records = load_records(line_json)
    overrides = [Path(path) for path in args.inputs] if args.inputs else []
    if overrides and len(overrides) != len(records):
        raise ValueError("--inputs count must match the number of records in --line-json")

    out_dir = Path(args.out_dir)
    used_names: set[str] = set()
    results = []
    for idx, record in enumerate(records):
        override = overrides[idx] if overrides else None
        image_path = image_path_for_record(record, line_json, override)
        results.append(
            process_record(
                record,
                line_json,
                image_path,
                out_dir,
                used_names,
                args.debug,
                args.line_thickness,
            )
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "apriltag_lines_overlay_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"line_json": str(line_json), "overlays": results}, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
