from __future__ import annotations

import cv2
import numpy as np

from court_geometry import LineCluster
from line_mask import line_region_for_cluster


def draw_roi_box(image: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    out = image.copy()
    if bbox is None:
        return out
    x1, y1, x2, y2 = bbox
    overlay = out.copy()
    overlay[:y1, :] = 0
    overlay[y2:, :] = 0
    overlay[:, :x1] = 0
    overlay[:, x2:] = 0
    out = cv2.addWeighted(overlay, 0.82, out, 0.18, 0)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(
        out,
        "roboflow court ROI",
        (x1 + 10, max(30, y1 - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    return out


def draw_all_clusters(image: np.ndarray, clusters: list[LineCluster]) -> np.ndarray:
    out = image.copy()
    colors = [
        (40, 40, 255),
        (255, 210, 0),
        (0, 220, 255),
        (80, 255, 80),
        (255, 80, 255),
        (0, 130, 255),
    ]
    for index, line in enumerate(clusters[:28]):
        color = colors[index % len(colors)]
        p1 = tuple(np.round(line.p1).astype(int))
        p2 = tuple(np.round(line.p2).astype(int))
        cv2.line(out, p1, p2, color, 3, cv2.LINE_AA)
        mx = int((p1[0] + p2[0]) / 2)
        my = int((p1[1] + p2[1]) / 2)
        cv2.putText(out, str(index), (mx + 4, my + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return out


def draw_selected(image: np.ndarray, selected: dict[str, LineCluster]) -> np.ndarray:
    out = image.copy()
    styles = {
        "baseline": ((0, 220, 255), 8),
        "single_line_left": ((40, 40, 255), 7),
        "single_line_right": ((40, 255, 40), 7),
        "near_baseline": ((0, 220, 255), 9),
        "near_left": ((40, 40, 255), 8),
        "near_right": ((40, 255, 40), 8),
    }
    for name, line in selected.items():
        color, width = styles[name]
        p1 = tuple(np.round(line.p1).astype(int))
        p2 = tuple(np.round(line.p2).astype(int))
        cv2.line(out, p1, p2, color, width, cv2.LINE_AA)
        mx = int((p1[0] + p2[0]) / 2)
        my = int((p1[1] + p2[1]) / 2)
        cv2.putText(out, name, (mx + 8, my - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def draw_colored_region_overlay(
    image: np.ndarray, mask: np.ndarray, selected: dict[str, LineCluster], grow_px: float
) -> np.ndarray:
    out = image.copy()
    colors = {
        "near_baseline": (0, 220, 255),
        "near_left": (40, 40, 255),
        "near_right": (40, 255, 40),
    }

    for name, line in selected.items():
        color = colors.get(name, (0, 220, 255))
        region = line_region_for_cluster(mask, line, grow_px)
        color_layer = np.zeros_like(image)
        color_layer[:] = color
        blended = cv2.addWeighted(out, 0.42, color_layer, 0.58, 0)
        out[region > 0] = blended[region > 0]

        p1 = tuple(np.round(line.p1).astype(int))
        p2 = tuple(np.round(line.p2).astype(int))
        cv2.line(out, p1, p2, color, 3, cv2.LINE_AA)

    return out


def draw_region_overlay(image: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    out = image.copy()
    color_layer = np.zeros_like(image)
    color_layer[:] = (0, 220, 255)
    blended = cv2.addWeighted(image, 0.45, color_layer, 0.55, 0)
    out[region_mask > 0] = blended[region_mask > 0]
    return out
