from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LineCluster:
    angle_deg: float
    rho: float
    weight: float
    support: int
    p1: tuple[float, float]
    p2: tuple[float, float]
    length: float


def normalize_angle(angle_deg: float) -> float:
    while angle_deg > 90.0:
        angle_deg -= 180.0
    while angle_deg <= -90.0:
        angle_deg += 180.0
    return angle_deg


def segment_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    return normalize_angle(math.degrees(math.atan2(y2 - y1, x2 - x1)))


def line_normal_from_angle(angle_deg: float) -> tuple[float, float]:
    theta = math.radians(angle_deg + 90.0)
    return math.cos(theta), math.sin(theta)


def canonical_line(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float, float]:
    angle = segment_angle(x1, y1, x2, y2)
    nx, ny = line_normal_from_angle(angle)
    rho = nx * x1 + ny * y1
    if rho < 0:
        rho = -rho
        angle = normalize_angle(angle + 180.0)
        nx, ny = -nx, -ny
    length = math.hypot(x2 - x1, y2 - y1)
    return angle, rho, nx, ny, length


def x_at_y(line: LineCluster, y: float) -> float | None:
    x1, y1 = line.p1
    x2, y2 = line.p2
    dy = y2 - y1
    if abs(dy) < 1e-6:
        return None
    t = (y - y1) / dy
    return x1 + t * (x2 - x1)


def bottom_x(line: LineCluster, image_h: int) -> float:
    x = x_at_y(line, image_h - 1)
    if x is not None:
        return x
    return (line.p1[0] + line.p2[0]) / 2.0


def point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def endpoint_distance(line: LineCluster, point: tuple[float, float]) -> float:
    return min(point_distance(line.p1, point), point_distance(line.p2, point))


def midpoint(line: LineCluster) -> tuple[float, float]:
    return ((line.p1[0] + line.p2[0]) / 2.0, (line.p1[1] + line.p2[1]) / 2.0)


def lower_endpoint(line: LineCluster) -> tuple[float, float]:
    return line.p1 if line.p1[1] >= line.p2[1] else line.p2


def baseline_endpoints(line: LineCluster) -> tuple[tuple[float, float], tuple[float, float]]:
    return (line.p1, line.p2) if line.p1[0] <= line.p2[0] else (line.p2, line.p1)


def serialize_line(line: LineCluster) -> dict:
    return {
        "angle_deg": round(line.angle_deg, 3),
        "rho": round(line.rho, 3),
        "support": line.support,
        "length": round(line.length, 3),
        "p1": [round(line.p1[0], 2), round(line.p1[1], 2)],
        "p2": [round(line.p2[0], 2), round(line.p2[1], 2)],
    }
