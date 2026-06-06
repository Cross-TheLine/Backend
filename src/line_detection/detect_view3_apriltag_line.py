from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np


APRILTAG_DICTIONARY = cv2.aruco.DICT_APRILTAG_36H11
MARKER_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))


def parse_marker_ids(value: str | None) -> list[int] | None:
    if not value:
        return None
    marker_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(marker_ids) != 2:
        raise ValueError("--line-marker-ids needs exactly two comma-separated marker IDs")
    return marker_ids


def tag_dictionary() -> cv2.aruco.Dictionary:
    return cv2.aruco.getPredefinedDictionary(APRILTAG_DICTIONARY)


def read_image_exif(path: Path) -> np.ndarray:
    try:
        from PIL import Image, ImageOps

        pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        return image


def rounded_point(point: list[float] | np.ndarray) -> list[float]:
    return [round(float(point[0]), 2), round(float(point[1]), 2)]


def detect_apriltags(image: np.ndarray, min_side_px: float) -> list[dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(tag_dictionary(), params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return []

    markers: list[dict] = []
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        pts = marker_corners.reshape(4, 2).astype(np.float32)
        side_lengths = [
            float(np.linalg.norm(pts[i] - pts[(i + 1) % 4]))
            for i in range(4)
        ]
        if min(side_lengths) < min_side_px:
            continue
        center = pts.mean(axis=0)
        markers.append(
            {
                "id": int(marker_id),
                "center": [float(center[0]), float(center[1])],
                "corners": [[float(x), float(y)] for x, y in pts],
                "side_px_mean": float(sum(side_lengths) / 4.0),
                "side_px_min": float(min(side_lengths)),
                "side_px_max": float(max(side_lengths)),
                "area_px": float(cv2.contourArea(pts)),
            }
        )

    markers.sort(key=lambda marker: (marker["id"], marker["center"][1], marker["center"][0]))
    for marker_index, marker in enumerate(markers, start=1):
        marker["marker_index"] = marker_index
    return markers


def marker_edges(marker: dict) -> list[tuple[tuple[int, int], np.ndarray]]:
    corners = np.array(marker["corners"], dtype=np.float32)
    return [(edge, corners[list(edge)]) for edge in MARKER_EDGES]


def fit_line_model(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    direction = np.array([vx, vy], dtype=np.float32)
    point_on_line = np.array([x0, y0], dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        raise ValueError("Cannot fit a line to degenerate points")
    return direction / norm, point_on_line


def line_fit_error(points: np.ndarray, direction: np.ndarray, point_on_line: np.ndarray) -> float:
    offsets = points.astype(np.float32) - point_on_line.astype(np.float32)
    cross = offsets[:, 0] * direction[1] - offsets[:, 1] * direction[0]
    return float(np.sqrt(np.mean(cross * cross)))


def point_line_distance(point: np.ndarray, direction: np.ndarray, point_on_line: np.ndarray) -> float:
    offset = point.astype(np.float32) - point_on_line.astype(np.float32)
    return abs(float(offset[0] * direction[1] - offset[1] * direction[0]))


def line_border_intersections(
    point_on_line: np.ndarray,
    direction: np.ndarray,
    width: int,
    height: int,
) -> list[list[float]] | None:
    x0, y0 = point_on_line.astype(float)
    dx, dy = direction.astype(float)
    candidates: list[tuple[float, float]] = []

    if abs(dx) > 1e-9:
        for x in (0.0, float(width - 1)):
            t = (x - x0) / dx
            y = y0 + t * dy
            if 0 <= y <= height - 1:
                candidates.append((x, y))
    if abs(dy) > 1e-9:
        for y in (0.0, float(height - 1)):
            t = (y - y0) / dy
            x = x0 + t * dx
            if 0 <= x <= width - 1:
                candidates.append((x, y))

    if len(candidates) < 2:
        return None

    best: tuple[tuple[float, float], tuple[float, float]] | None = None
    best_dist = -1.0
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            dist = float(np.linalg.norm(np.array(candidates[i]) - np.array(candidates[j])))
            if dist > best_dist:
                best_dist = dist
                best = (candidates[i], candidates[j])
    if best is None:
        return None
    return [[float(best[0][0]), float(best[0][1])], [float(best[1][0]), float(best[1][1])]]


def ray_endpoints(
    border_points: list[list[float]] | None,
    anchor_point: np.ndarray,
    through_point: np.ndarray,
) -> list[list[float]] | None:
    if border_points is None:
        return None
    direction = through_point.astype(np.float32) - anchor_point.astype(np.float32)
    if float(np.linalg.norm(direction)) < 1e-9:
        return None
    border = max(
        border_points,
        key=lambda point: float(np.dot(np.array(point, dtype=np.float32) - anchor_point, direction)),
    )
    return [rounded_point(border), rounded_point(anchor_point)]


def select_view3_roles(
    markers: list[dict],
    line_marker_ids: list[int] | None,
) -> tuple[dict[str, dict], float]:
    if len(markers) < 3:
        raise RuntimeError(f"Need at least 3 AprilTags for view3, found {len(markers)}")

    picked = sorted(markers, key=lambda marker: marker["area_px"], reverse=True)[:3]
    if line_marker_ids is not None:
        marker_by_id = {marker["id"]: marker for marker in markers}
        missing = [marker_id for marker_id in line_marker_ids if marker_id not in marker_by_id]
        if missing:
            raise RuntimeError(f"Could not find requested line marker IDs: {missing}")
        line_pair = (marker_by_id[line_marker_ids[0]], marker_by_id[line_marker_ids[1]])
        inside_candidates = [marker for marker in markers if marker["id"] not in line_marker_ids]
        if not inside_candidates:
            raise RuntimeError("Need a third marker outside --line-marker-ids to decide the outer edge")
        inside_marker = max(inside_candidates, key=lambda marker: marker["area_px"])
    else:
        best_pair: tuple[dict, dict] | None = None
        best_score = float("inf")
        for marker_a, marker_b in combinations(picked, 2):
            center_a = np.array(marker_a["center"], dtype=np.float32)
            center_b = np.array(marker_b["center"], dtype=np.float32)
            dx = abs(float(center_a[0] - center_b[0]))
            dy = abs(float(center_a[1] - center_b[1]))
            vertical_score = dx / (dy + 1.0)
            if vertical_score < best_score:
                best_score = vertical_score
                best_pair = (marker_a, marker_b)
        if best_pair is None:
            raise RuntimeError("Could not select view3 line markers")
        line_pair = best_pair
        inside_marker = next(marker for marker in picked if marker not in line_pair)

    center_a = np.array(line_pair[0]["center"], dtype=np.float32)
    center_b = np.array(line_pair[1]["center"], dtype=np.float32)
    vertical_score = abs(float(center_a[0] - center_b[0])) / (abs(float(center_a[1] - center_b[1])) + 1.0)
    line_top, line_bottom = sorted(line_pair, key=lambda marker: marker["center"][1])
    return {
        "line_marker_top": line_top,
        "line_marker_bottom": line_bottom,
        "inside_marker": inside_marker,
    }, vertical_score


def select_outer_edge_line(
    line_top: dict,
    line_bottom: dict,
    inside_marker: dict,
    vertical_weight: float,
) -> dict:
    inside_center = np.array(inside_marker["center"], dtype=np.float32)
    candidates: list[dict] = []
    for top_edge, top_points in marker_edges(line_top):
        for bottom_edge, bottom_points in marker_edges(line_bottom):
            fit_points = np.vstack([top_points, bottom_points])
            direction, point_on_line = fit_line_model(fit_points)
            fit_error = line_fit_error(fit_points, direction, point_on_line)
            vertical_error = abs(float(direction[0]))
            inside_distance = point_line_distance(inside_center, direction, point_on_line)
            geom_score = fit_error + vertical_error * vertical_weight
            candidates.append(
                {
                    "top_edge": top_edge,
                    "bottom_edge": bottom_edge,
                    "top_points": top_points,
                    "bottom_points": bottom_points,
                    "fit_points": fit_points,
                    "direction": direction,
                    "point_on_line": point_on_line,
                    "fit_error": fit_error,
                    "vertical_error": vertical_error,
                    "inside_distance": inside_distance,
                    "geom_score": geom_score,
                }
            )

    best_score = min(candidate["geom_score"] for candidate in candidates)
    viable = [candidate for candidate in candidates if candidate["geom_score"] <= best_score + 3.0]
    return max(viable, key=lambda candidate: candidate["inside_distance"])


def build_view3_line(
    image_shape: tuple[int, int, int],
    roles: dict[str, dict],
    vertical_weight: float,
) -> dict:
    height, width = image_shape[:2]
    line_top = roles["line_marker_top"]
    line_bottom = roles["line_marker_bottom"]
    inside_marker = roles["inside_marker"]
    selected = select_outer_edge_line(line_top, line_bottom, inside_marker, vertical_weight)

    top_endpoint = selected["top_points"].mean(axis=0)
    bottom_endpoint = selected["bottom_points"].mean(axis=0)
    border = line_border_intersections(selected["point_on_line"], selected["direction"], width, height)
    extended = ray_endpoints(border, bottom_endpoint, top_endpoint)

    return {
        "name": "view3_vertical_outer_line",
        "points": extended or [rounded_point(top_endpoint), rounded_point(bottom_endpoint)],
    }


def process_image(
    path: Path,
    min_side_px: float,
    line_marker_ids: list[int] | None,
    vertical_weight: float,
) -> dict:
    image = read_image_exif(path)
    height, width = image.shape[:2]
    markers = detect_apriltags(image, min_side_px)
    roles, _ = select_view3_roles(markers, line_marker_ids)
    line = build_view3_line(image.shape, roles, vertical_weight)
    return {
        "image": str(path),
        "width": width,
        "height": height,
        "view": "view3",
        "lines": [line],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect one screen-vertical AprilTag-guided view3 line and export JSON.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--min-side-px", type=float, default=0.0)
    parser.add_argument(
        "--line-marker-ids",
        default=None,
        help="Optional two marker IDs that lie on the target line, e.g. 1,2.",
    )
    parser.add_argument(
        "--vertical-weight",
        type=float,
        default=30.0,
        help="Penalty weight for edge candidates that are not screen-vertical.",
    )
    args = parser.parse_args()

    line_marker_ids = parse_marker_ids(args.line_marker_ids)
    records = [
        process_image(Path(input_path), args.min_side_px, line_marker_ids, args.vertical_weight)
        for input_path in args.inputs
    ]
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
