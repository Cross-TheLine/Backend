from __future__ import annotations

import math

import cv2
import numpy as np

from court_geometry import LineCluster, canonical_line, line_normal_from_angle


def hough_segments(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=58,
        minLineLength=70,
        maxLineGap=24,
    )
    if raw is None:
        return []
    return [tuple(map(int, line[0])) for line in raw]


def cluster_segments(segments: list[tuple[int, int, int, int]], mask: np.ndarray) -> list[LineCluster]:
    buckets: list[list[tuple[float, float, float, tuple[int, int, int, int]]]] = []
    for x1, y1, x2, y2 in segments:
        angle, rho, _, _, length = canonical_line(x1, y1, x2, y2)
        if length < 70:
            continue
        placed = False
        for bucket in buckets:
            b_angle = np.average([item[0] for item in bucket], weights=[item[2] for item in bucket])
            b_rho = np.average([item[1] for item in bucket], weights=[item[2] for item in bucket])
            if abs(angle - b_angle) < 5.0 and abs(rho - b_rho) < 24.0:
                bucket.append((angle, rho, length, (x1, y1, x2, y2)))
                placed = True
                break
        if not placed:
            buckets.append([(angle, rho, length, (x1, y1, x2, y2))])

    clusters: list[LineCluster] = []
    ys, xs = np.nonzero(mask)
    for bucket in buckets:
        weights = np.array([item[2] for item in bucket], dtype=np.float64)
        angle = float(np.average([item[0] for item in bucket], weights=weights))
        rho = float(np.average([item[1] for item in bucket], weights=weights))
        nx, ny = line_normal_from_angle(angle)
        distances = np.abs(xs * nx + ys * ny - rho)
        support_mask = distances < 8.0
        support = int(support_mask.sum())
        if support < 95:
            continue

        direction = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
        sx = xs[support_mask]
        sy = ys[support_mask]
        t = sx * direction[0] + sy * direction[1]
        lo = float(np.percentile(t, 2))
        hi = float(np.percentile(t, 98))
        length = hi - lo
        if length < 95:
            continue
        base = np.array([nx * rho, ny * rho])
        p1 = base + direction * lo
        p2 = base + direction * hi
        clusters.append(
            LineCluster(
                angle_deg=angle,
                rho=rho,
                weight=float(weights.sum()),
                support=support,
                p1=(float(p1[0]), float(p1[1])),
                p2=(float(p2[0]), float(p2[1])),
                length=float(length),
            )
        )

    clusters.sort(key=lambda c: c.length * math.sqrt(c.support), reverse=True)
    return merge_duplicate_clusters(clusters)


def merge_duplicate_clusters(clusters: list[LineCluster]) -> list[LineCluster]:
    kept: list[LineCluster] = []
    for cluster in clusters:
        duplicate = False
        for old in kept:
            if abs(cluster.angle_deg - old.angle_deg) < 7.5 and abs(cluster.rho - old.rho) < 42.0:
                duplicate = True
                break
        if not duplicate:
            kept.append(cluster)
    return kept
