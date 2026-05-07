from __future__ import annotations

import math

import cv2
import numpy as np

from court_geometry import LineCluster, line_normal_from_angle


def make_white_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    b, g, r = cv2.split(image)
    spread = np.maximum.reduce([b, g, r]) - np.minimum.reduce([b, g, r])

    mask = (
        (v > 185)
        & (s < 58)
        & (spread < 55)
        & (b > 165)
        & (g > 165)
        & (r > 165)
        & (np.abs(r.astype(np.int16) - g.astype(np.int16)) < 46)
        & (np.abs(g.astype(np.int16) - b.astype(np.int16)) < 46)
        & (np.abs(r.astype(np.int16) - b.astype(np.int16)) < 46)
    ).astype(np.uint8) * 255

    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
    mask = keep_line_like_components(mask)
    return mask


def keep_line_like_components(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    image_area = mask.shape[0] * mask.shape[1]

    for label in range(1, num_labels):
        _, _, width, height, area = stats[label]
        if area < 35:
            continue

        bbox_area = max(1, width * height)
        fill_ratio = area / bbox_area
        large_filled_region = area > image_area * 0.025 and fill_ratio > 0.28
        thick_patch = min(width, height) > 70 and fill_ratio > 0.35

        if large_filled_region or thick_patch:
            continue

        filtered[labels == label] = 255

    return filtered


def line_region_for_cluster(mask: np.ndarray, line: LineCluster, grow_px: float) -> np.ndarray:
    region = np.zeros_like(mask)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return region

    direction = np.array(
        [math.cos(math.radians(line.angle_deg)), math.sin(math.radians(line.angle_deg))],
        dtype=np.float64,
    )
    p1 = np.array(line.p1, dtype=np.float64)
    p2 = np.array(line.p2, dtype=np.float64)
    t1 = float(p1 @ direction)
    t2 = float(p2 @ direction)
    t_lo = min(t1, t2)
    t_hi = max(t1, t2)
    endpoint_margin = max(18.0, grow_px * 2.0, line.length * 0.025)

    nx, ny = line_normal_from_angle(line.angle_deg)
    distances = np.abs(xs * nx + ys * ny - line.rho)
    positions = xs * direction[0] + ys * direction[1]
    along_segment = (positions >= t_lo - endpoint_margin) & (positions <= t_hi + endpoint_margin)

    candidate_pixels = (distances <= grow_px) & along_segment
    if not np.any(candidate_pixels):
        return region

    candidate = np.zeros_like(mask)
    candidate[ys[candidate_pixels], xs[candidate_pixels]] = 255

    seed_px = max(4.0, grow_px * 0.35)
    seed_pixels = (distances <= seed_px) & along_segment
    if not np.any(seed_pixels):
        return candidate

    seed = np.zeros_like(mask)
    seed[ys[seed_pixels], xs[seed_pixels]] = 255
    _, labels = cv2.connectedComponents(candidate, connectivity=8)
    seed_labels = np.unique(labels[seed > 0])

    for label in seed_labels:
        if label != 0:
            region[labels == label] = 255
    return region


def line_region_mask(mask: np.ndarray, lines: list[LineCluster], grow_px: float) -> np.ndarray:
    region = np.zeros_like(mask)
    for line in lines:
        region = cv2.bitwise_or(region, line_region_for_cluster(mask, line, grow_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(region, cv2.MORPH_CLOSE, kernel, iterations=1)
