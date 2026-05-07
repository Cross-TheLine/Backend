from __future__ import annotations

import math

from court_geometry import (
    LineCluster,
    baseline_endpoints,
    bottom_x,
    endpoint_distance,
    lower_endpoint,
    midpoint,
)


def classify_near_three_lines(clusters: list[LineCluster], image_shape: tuple[int, int, int]) -> dict[str, LineCluster]:
    h, w = image_shape[:2]
    min_length = min(w, h) * 0.28
    near_candidates = [
        line
        for line in clusters
        if line.length >= min_length
        and max(line.p1[1], line.p2[1]) >= h * 0.48
        and midpoint(line)[1] >= h * 0.38
    ]
    if len(near_candidates) < 3:
        raise RuntimeError("가까운 쪽 3줄 후보가 부족합니다.")

    best_pair: tuple[LineCluster, LineCluster] | None = None
    best_pair_score = -float("inf")
    for i, first in enumerate(near_candidates):
        for second in near_candidates[i + 1 :]:
            angle_gap = abs(first.angle_deg - second.angle_deg)
            if angle_gap > 14.0:
                continue

            first_anchor = lower_endpoint(first)
            second_anchor = lower_endpoint(second)
            anchor_y_gap = abs(first_anchor[1] - second_anchor[1])
            if min(first_anchor[1], second_anchor[1]) < h * 0.62 or anchor_y_gap > h * 0.12:
                continue

            separation = abs(first_anchor[0] - second_anchor[0])
            if separation < w * 0.055:
                continue

            bottom_bonus = (first_anchor[1] + second_anchor[1]) / h
            support_score = math.sqrt(first.support) + math.sqrt(second.support)
            pair_score = (
                first.length
                + second.length
                + support_score * 10.0
                + bottom_bonus * 820.0
                - angle_gap * 24.0
                - anchor_y_gap * 1.6
            )
            if pair_score > best_pair_score:
                best_pair = (first, second)
                best_pair_score = pair_score

    if best_pair is None:
        raise RuntimeError("가까운 쪽 좌/우 경계선 후보 2개를 찾지 못했습니다.")

    side_a, side_b = best_pair
    side_anchors = [lower_endpoint(side_a), lower_endpoint(side_b)]
    baseline_candidates = [
        line
        for line in near_candidates
        if line is not side_a
        and line is not side_b
        and abs(line.angle_deg - ((side_a.angle_deg + side_b.angle_deg) / 2.0)) >= 18.0
        and max(line.p1[1], line.p2[1]) >= h * 0.48
    ]
    if not baseline_candidates:
        baseline_candidates = [line for line in near_candidates if line is not side_a and line is not side_b]

    def baseline_score(line: LineCluster) -> float:
        close_to_sides = sum(endpoint_distance(line, anchor) for anchor in side_anchors)
        lower = max(line.p1[1], line.p2[1]) / h
        support_score = math.sqrt(line.support)
        return line.length + support_score * 10.0 + lower * 420.0 - close_to_sides * 2.4

    baseline = max(baseline_candidates, key=baseline_score)
    sides = [side_a, side_b]
    sides.sort(key=lambda line: lower_endpoint(line)[0])

    return {
        "near_baseline": baseline,
        "near_left": sides[0],
        "near_right": sides[1],
    }


def classify_lines(clusters: list[LineCluster], image_shape: tuple[int, int, int]) -> dict[str, LineCluster]:
    h, w = image_shape[:2]

    baseline_candidates = [
        line
        for line in clusters
        if 5.0 <= line.angle_deg <= 35.0
        and max(line.p1[1], line.p2[1]) > h * 0.72
        and line.length > w * 0.28
    ]
    if not baseline_candidates:
        raise RuntimeError("baseline 후보를 찾지 못했습니다.")

    baseline = max(
        baseline_candidates,
        key=lambda line: line.length * math.sqrt(line.support) + max(line.p1[1], line.p2[1]) * 4.0,
    )

    singles_candidates = [
        line
        for line in clusters
        if -78.0 <= line.angle_deg <= -25.0
        and line.length > h * 0.45
        and max(line.p1[1], line.p2[1]) > h * 0.76
    ]
    singles_candidates.sort(key=lambda line: line.length * math.sqrt(line.support), reverse=True)

    singles: list[LineCluster] = []
    for candidate in singles_candidates:
        bx = bottom_x(candidate, h)
        separated = all(abs(bx - bottom_x(old, h)) > w * 0.18 for old in singles)
        if separated:
            singles.append(candidate)
        if len(singles) == 2:
            break

    if len(singles) < 2:
        left_endpoint, right_endpoint = baseline_endpoints(baseline)
        join_radius = max(85.0, w * 0.08)
        side_candidates = [
            line
            for line in clusters
            if line is not baseline
            and -80.0 <= line.angle_deg <= 12.0
            and line.length > min(w, h) * 0.48
            and max(line.p1[1], line.p2[1]) > h * 0.56
        ]

        left_joined = [
            line
            for line in side_candidates
            if endpoint_distance(line, left_endpoint) <= join_radius
        ]
        right_joined = [
            line
            for line in side_candidates
            if endpoint_distance(line, right_endpoint) <= join_radius
        ]

        if left_joined and right_joined:
            left_line = max(
                left_joined,
                key=lambda line: line.length * math.sqrt(line.support)
                - endpoint_distance(line, left_endpoint) * 15.0,
            )
            right_line = max(
                right_joined,
                key=lambda line: line.length * math.sqrt(line.support)
                - endpoint_distance(line, right_endpoint) * 15.0,
            )
            singles = [left_line, right_line]

    if len(singles) < 2:
        raise RuntimeError("싱글라인 후보 2개를 찾지 못했습니다.")

    singles.sort(key=lambda line: bottom_x(line, h))
    return {
        "baseline": baseline,
        "single_line_left": singles[0],
        "single_line_right": singles[1],
    }
