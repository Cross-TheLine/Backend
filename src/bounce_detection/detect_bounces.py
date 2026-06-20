from __future__ import annotations

import csv

import cv2


def point_at(track, idx):
    if idx < 0 or idx >= len(track):
        return None
    x, y = track[idx]
    if x is None or y is None:
        return None
    return float(x), float(y)


def detect_y_reversal_bounces(track, fps, args):
    rows = []
    radius = args.window
    local_radius = args.local_window

    for idx in range(radius, len(track) - radius):
        center = point_at(track, idx)
        before = point_at(track, idx - radius)
        after = point_at(track, idx + radius)
        if center is None or before is None or after is None:
            continue

        x, y = center
        prev_x, prev_y = before
        next_x, next_y = after
        vy_before = (y - prev_y) / radius
        vy_after = (next_y - y) / radius
        vx_before = (x - prev_x) / radius
        vx_after = (next_x - x) / radius

        if vy_before < args.min_down_speed:
            continue
        if vy_after > -args.min_up_speed:
            continue

        local_points = [
            point_at(track, local_idx)
            for local_idx in range(idx - local_radius, idx + local_radius + 1)
        ]
        local_points = [point for point in local_points if point is not None]
        if len(local_points) < args.min_local_points:
            continue

        max_y = max(point[1] for point in local_points)
        if y < max_y - args.local_y_tolerance:
            continue

        x_velocity_change = abs(vx_after - vx_before)
        if x_velocity_change > args.max_x_velocity_change:
            continue

        prominence = y - min(prev_y, next_y)
        score = (vy_before - vy_after) + max(prominence, 0.0) * 0.35 - x_velocity_change * 0.12
        if score < args.min_score:
            continue

        rows.append({
            "frame_index": idx,
            "time_sec": idx / fps if fps else 0.0,
            "x": x,
            "y": y,
            "score": score,
            "vy_before": vy_before,
            "vy_after": vy_after,
            "prominence": prominence,
            "x_velocity_change": x_velocity_change,
        })

    rows.sort(key=lambda row: row["score"], reverse=True)
    selected = []
    for row in rows:
        if any(abs(row["frame_index"] - kept["frame_index"]) < args.min_gap for kept in selected):
            continue
        selected.append(row)

    return sorted(selected, key=lambda row: row["frame_index"])


def write_track_csv(path, track, statuses, scores, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["frame_index", "time_sec", "x", "y", "status", "score"],
        )
        writer.writeheader()
        for idx, (x, y) in enumerate(track):
            writer.writerow({
                "frame_index": idx,
                "time_sec": round(idx / fps, 4) if fps else 0.0,
                "x": "" if x is None else round(float(x), 2),
                "y": "" if y is None else round(float(y), 2),
                "status": statuses[idx] if statuses and idx < len(statuses) else "",
                "score": "" if scores is None or idx >= len(scores) else round(float(scores[idx]), 4),
            })


def write_bounce_csv(path, bounces):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "frame_index", "time_sec", "x", "y", "score",
                "vy_before", "vy_after", "prominence", "x_velocity_change",
            ],
        )
        writer.writeheader()
        for row in bounces:
            writer.writerow({
                "frame_index": int(row["frame_index"]),
                "time_sec": round(float(row["time_sec"]), 4),
                "x": round(float(row["x"]), 2),
                "y": round(float(row["y"]), 2),
                "score": round(float(row["score"]), 4),
                "vy_before": round(float(row["vy_before"]), 4),
                "vy_after": round(float(row["vy_after"]), 4),
                "prominence": round(float(row["prominence"]), 4),
                "x_velocity_change": round(float(row["x_velocity_change"]), 4),
            })


def write_video(frames, track, bounces, output_path, fps):
    if not frames:
        return
    height, width = frames[0].shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"DIVX"),
        fps,
        (width, height),
    )
    bounce_by_frame = {int(row["frame_index"]): row for row in bounces}

    for idx, frame in enumerate(frames):
        canvas = frame.copy()
        for offset in range(8):
            track_idx = idx - offset
            point = point_at(track, track_idx)
            if point is None:
                continue
            x, y = point
            alpha = 1.0 - offset / 8.0
            color = (0, int(80 + 175 * alpha), 255)
            cv2.circle(canvas, (int(x), int(y)), 5, color, -1)

        visible = None
        for delta in range(4):
            visible = bounce_by_frame.get(idx - delta) or bounce_by_frame.get(idx + delta)
            if visible is not None:
                break
        if visible is not None:
            x = int(visible["x"])
            y = int(visible["y"])
            cv2.circle(canvas, (x, y), 18, (0, 255, 255), 3)
            cv2.putText(
                canvas,
                "Y-BOUNCE",
                (max(0, x - 45), max(24, y - 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        writer.write(canvas)
    writer.release()
