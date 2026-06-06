from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


APRILTAG_FAMILIES = {
    "4x4_1000": cv2.aruco.DICT_4X4_1000,
    "aruco_mip_36h12": cv2.aruco.DICT_ARUCO_MIP_36H12,
    "tag16h5": cv2.aruco.DICT_APRILTAG_16H5,
    "tag25h9": cv2.aruco.DICT_APRILTAG_25H9,
    "tag36h10": cv2.aruco.DICT_APRILTAG_36H10,
    "tag36h11": cv2.aruco.DICT_APRILTAG_36H11,
}

MARKER_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))


def family_dictionary(family: str) -> int:
    key = family.lower()
    if key not in APRILTAG_FAMILIES:
        known = ", ".join(sorted(APRILTAG_FAMILIES))
        raise ValueError(f"Unknown AprilTag family {family!r}. Known: {known}")
    return APRILTAG_FAMILIES[key]


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


def detect_apriltags(image: np.ndarray, family: str, min_side_px: float) -> tuple[list[dict], dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(family_dictionary(family))
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, rejected = detector.detectMarkers(gray)
    if ids is None:
        return [], {
            "raw_marker_count": 0,
            "rejected_count": 0 if rejected is None else len(rejected),
        }

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
    return markers, {
        "raw_marker_count": len(ids),
        "rejected_count": 0 if rejected is None else len(rejected),
    }


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


def marker_summary(marker: dict) -> dict:
    return {
        "marker_index": marker["marker_index"],
        "id": marker["id"],
        "center": rounded_point(marker["center"]),
        "side_px_mean": round(float(marker["side_px_mean"]), 2),
    }


def assign_view2_roles(markers: list[dict]) -> tuple[dict[str, dict], str]:
    if len(markers) < 3:
        raise RuntimeError(f"Need at least 3 AprilTags for view2, found {len(markers)}")

    picked = sorted(markers, key=lambda marker: marker["area_px"], reverse=True)[:3]
    vertex = max(picked, key=lambda marker: marker["center"][1])
    remaining = [marker for marker in picked if marker is not vertex]
    top_marker = min(remaining, key=lambda marker: marker["center"][1])
    side_marker = max(remaining, key=lambda marker: marker["center"][1])
    layout = "left" if top_marker["center"][0] < side_marker["center"][0] else "right"
    return {
        "top_marker": top_marker,
        "side_marker": side_marker,
        "vertex": vertex,
    }, layout


def select_outer_edge_pair(
    line_marker: dict,
    vertex_marker: dict,
    inside_marker: dict,
) -> dict:
    line_center = np.array(line_marker["center"], dtype=np.float32)
    vertex_center = np.array(vertex_marker["center"], dtype=np.float32)
    center_direction = vertex_center - line_center
    center_norm = float(np.linalg.norm(center_direction))
    if center_norm < 1e-9:
        center_direction = np.array([0.0, 1.0], dtype=np.float32)
    else:
        center_direction = center_direction / center_norm

    inside_center = np.array(inside_marker["center"], dtype=np.float32)
    candidates: list[dict] = []
    for line_edge, line_points in marker_edges(line_marker):
        for vertex_edge, vertex_points in marker_edges(vertex_marker):
            fit_points = np.vstack([line_points, vertex_points])
            direction, point_on_line = fit_line_model(fit_points)
            fit_error = line_fit_error(fit_points, direction, point_on_line)
            parallel_error = abs(float(direction[0] * center_direction[1] - direction[1] * center_direction[0]))
            geom_score = fit_error + parallel_error * 20.0
            inside_distance = point_line_distance(inside_center, direction, point_on_line)
            candidates.append(
                {
                    "line_edge": line_edge,
                    "vertex_edge": vertex_edge,
                    "line_points": line_points,
                    "vertex_points": vertex_points,
                    "fit_points": fit_points,
                    "direction": direction,
                    "point_on_line": point_on_line,
                    "fit_error": fit_error,
                    "parallel_error": parallel_error,
                    "geom_score": geom_score,
                    "inside_distance": inside_distance,
                }
            )

    best_score = min(candidate["geom_score"] for candidate in candidates)
    viable = [candidate for candidate in candidates if candidate["geom_score"] <= best_score + 3.0]
    return max(viable, key=lambda candidate: candidate["inside_distance"])


def build_view2_lines(image_shape: tuple[int, int, int], roles: dict[str, dict], layout: str) -> list[dict]:
    height, width = image_shape[:2]
    specs = [
        ("top_to_vertex", "top_marker", "side_marker"),
        ("side_to_vertex", "side_marker", "top_marker"),
    ]
    lines: list[dict] = []
    for name, line_role, inside_role in specs:
        line_marker = roles[line_role]
        vertex_marker = roles["vertex"]
        inside_marker = roles[inside_role]
        selected = select_outer_edge_pair(line_marker, vertex_marker, inside_marker)

        line_endpoint = selected["line_points"].mean(axis=0)
        vertex_endpoint = selected["vertex_points"].mean(axis=0)
        border = line_border_intersections(selected["point_on_line"], selected["direction"], width, height)
        extended = ray_endpoints(border, vertex_endpoint, line_endpoint)

        lines.append(
            {
                "name": name,
                "method": "view2_edge_fit",
                "layout": layout,
                "marker_roles": [line_role, "vertex"],
                "marker_indices": [line_marker["marker_index"], vertex_marker["marker_index"]],
                "marker_ids": [line_marker["id"], vertex_marker["id"]],
                "inside_marker_role": inside_role,
                "inside_marker_index": inside_marker["marker_index"],
                "inside_marker_id": inside_marker["id"],
                "edges": [
                    {
                        "marker_role": line_role,
                        "marker_index": line_marker["marker_index"],
                        "marker_id": line_marker["id"],
                        "edge": [int(selected["line_edge"][0]), int(selected["line_edge"][1])],
                    },
                    {
                        "marker_role": "vertex",
                        "marker_index": vertex_marker["marker_index"],
                        "marker_id": vertex_marker["id"],
                        "edge": [int(selected["vertex_edge"][0]), int(selected["vertex_edge"][1])],
                    },
                ],
                "fit_points": [rounded_point(point) for point in selected["fit_points"]],
                "segment_endpoints": [rounded_point(line_endpoint), rounded_point(vertex_endpoint)],
                "vertex_endpoint": rounded_point(vertex_endpoint),
                "extended_endpoints": extended,
                "fit_error_px": round(float(selected["fit_error"]), 3),
                "parallel_error": round(float(selected["parallel_error"]), 5),
                "inside_distance_px": round(float(selected["inside_distance"]), 3),
            }
        )
    return lines


def process_image(path: Path, family: str, min_side_px: float) -> dict:
    image = read_image_exif(path)
    markers, detection = detect_apriltags(image, family, min_side_px)
    roles, layout = assign_view2_roles(markers)
    lines = build_view2_lines(image.shape, roles, layout)
    return {
        "schema": "apriltag_view2_lines.v1",
        "image": str(path),
        "image_shape_hwc": list(image.shape),
        "family": family,
        "mode": "view2",
        "method": "view2_edge_fit",
        "marker_count": len(markers),
        "raw_marker_count": detection["raw_marker_count"],
        "rejected_count": detection["rejected_count"],
        "min_side_px": min_side_px,
        "layout": layout,
        "role_rule": "Use the largest three detected tags. The marker with the largest image y center is the vertex; the two remaining markers decide left/right layout.",
        "markers": markers,
        "roles": {role_name: marker_summary(marker) for role_name, marker in roles.items()},
        "lines": lines,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect AprilTag-guided view2 court-corner lines and export JSON.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--family", default="tag36h11", choices=sorted(APRILTAG_FAMILIES))
    parser.add_argument("--min-side-px", type=float, default=0.0)
    args = parser.parse_args()

    records = [
        process_image(Path(input_path), args.family, args.min_side_px)
        for input_path in args.inputs
    ]
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
