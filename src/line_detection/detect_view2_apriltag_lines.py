from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


APRILTAG_FAMILIES = {
    "tag36h11": cv2.aruco.DICT_APRILTAG_36H11,
    "apriltag_36h11": cv2.aruco.DICT_APRILTAG_36H11,
    "aruco_mip_36h12": cv2.aruco.DICT_ARUCO_MIP_36H12,
}
APRILTAG_DICTIONARY = APRILTAG_FAMILIES["tag36h11"]
MARKER_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))


def family_dictionary(family: str) -> int:
    normalized = family.lower()
    if normalized not in APRILTAG_FAMILIES:
        supported = ", ".join(sorted(APRILTAG_FAMILIES))
        raise ValueError(f"Unknown tag family: {family}. Supported: {supported}")
    return APRILTAG_FAMILIES[normalized]


def tag_dictionary(family: str = "tag36h11") -> cv2.aruco.Dictionary:
    return cv2.aruco.getPredefinedDictionary(family_dictionary(family))


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


def rounded_point(point: list[float] | np.ndarray) -> list[float]:
    return [round(float(point[0]), 2), round(float(point[1]), 2)]


def detect_apriltags(image: np.ndarray, min_side_px: float, family: str = "tag36h11") -> list[dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(tag_dictionary(family), params)
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


def fitted_line_intersection(first: dict, second: dict) -> np.ndarray | None:
    p1 = first["point_on_line"].astype(np.float32)
    d1 = first["direction"].astype(np.float32)
    p2 = second["point_on_line"].astype(np.float32)
    d2 = second["direction"].astype(np.float32)
    denom = float(d1[0] * d2[1] - d1[1] * d2[0])
    if abs(denom) < 1e-9:
        return None
    delta = p2 - p1
    t = float((delta[0] * d2[1] - delta[1] * d2[0]) / denom)
    return p1 + d1 * t


def shared_vertex_point(vertex_marker: dict, selections: list[dict], image_shape: tuple[int, int, int]) -> np.ndarray:
    height, width = image_shape[:2]
    corners = np.array(vertex_marker["corners"], dtype=np.float32)
    edge_sets = [set(selection["vertex_edge"]) for selection in selections]
    common = set.intersection(*edge_sets) if edge_sets else set()
    if len(common) == 1:
        return corners[next(iter(common))]

    if len(selections) >= 2:
        intersection = fitted_line_intersection(selections[0], selections[1])
        if intersection is not None:
            padding = max(width, height) * 0.08
            if -padding <= intersection[0] <= width - 1 + padding and -padding <= intersection[1] <= height - 1 + padding:
                return intersection

    endpoints = [selection["vertex_points"].mean(axis=0) for selection in selections]
    return np.mean(np.vstack(endpoints), axis=0)


def assign_view2_roles(markers: list[dict]) -> tuple[dict[str, dict], str]:
    if len(markers) < 3:
        raise RuntimeError(f"Need at least 3 AprilTags for view2, found {len(markers)}")

    picked = sorted(markers, key=lambda marker: marker["area_px"], reverse=True)[:3]
    vertex = max(picked, key=lambda marker: marker["center"][1])
    remaining = [marker for marker in picked if marker is not vertex]
    top_marker = min(remaining, key=lambda marker: marker["center"][1])
    side_marker = max(remaining, key=lambda marker: marker["center"][1])
    view_side = "left" if top_marker["center"][0] < side_marker["center"][0] else "right"
    return {
        "top_marker": top_marker,
        "side_marker": side_marker,
        "vertex": vertex,
    }, view_side


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


def build_view2_lines(image_shape: tuple[int, int, int], roles: dict[str, dict]) -> tuple[list[dict], list[float]]:
    height, width = image_shape[:2]
    specs = [
        ("top_to_vertex", "top_marker", "side_marker"),
        ("side_to_vertex", "side_marker", "top_marker"),
    ]

    selected_lines = []
    for name, line_role, inside_role in specs:
        line_marker = roles[line_role]
        vertex_marker = roles["vertex"]
        inside_marker = roles[inside_role]
        selected = select_outer_edge_pair(line_marker, vertex_marker, inside_marker)
        selected_lines.append((name, line_marker, selected))

    vertex_point = shared_vertex_point(
        roles["vertex"],
        [selected for _, _, selected in selected_lines],
        image_shape,
    )

    lines: list[dict] = []
    for name, line_marker, selected in selected_lines:
        line_endpoint = selected["line_points"].mean(axis=0)
        border = line_border_intersections(selected["point_on_line"], selected["direction"], width, height)
        extended = ray_endpoints(border, vertex_point, line_endpoint)

        lines.append(
            {
                "name": name,
                "points": extended or [rounded_point(line_endpoint), rounded_point(vertex_point)],
            }
        )
    return lines, rounded_point(vertex_point)


def process_image(path: Path, family: str = "tag36h11", min_side_px: float = 0.0) -> dict:
    image = read_image_exif(path)
    height, width = image.shape[:2]
    markers = detect_apriltags(image, min_side_px, family)
    roles, view_side = assign_view2_roles(markers)
    lines, vertex_point = build_view2_lines(image.shape, roles)
    return {
        "schema": "apriltag_lines.v1",
        "image": str(path),
        "width": width,
        "height": height,
        "mode": "view2",
        "view": "view2",
        "family": family,
        "view_side": view_side,
        "vertex_point": vertex_point,
        "marker_count": len(markers),
        "markers": markers,
        "lines": lines,
    }

