import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from src.bounce_detection.detect_bounces import (
    load_model,
    read_video,
    scale_track_to_frame,
    select_device,
    track_ball,
)


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
            'frame_index': idx,
            'time_sec': idx / fps if fps else 0.0,
            'x': x,
            'y': y,
            'score': score,
            'vy_before': vy_before,
            'vy_after': vy_after,
            'prominence': prominence,
            'x_velocity_change': x_velocity_change,
        })

    rows.sort(key=lambda row: row['score'], reverse=True)
    selected = []
    for row in rows:
        if any(abs(row['frame_index'] - kept['frame_index']) < args.min_gap for kept in selected):
            continue
        selected.append(row)

    return sorted(selected, key=lambda row: row['frame_index'])


def write_track_csv(path, track, statuses, scores, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=['frame_index', 'time_sec', 'x', 'y', 'status', 'score'],
        )
        writer.writeheader()
        for idx, (x, y) in enumerate(track):
            writer.writerow({
                'frame_index': idx,
                'time_sec': round(idx / fps, 4) if fps else 0.0,
                'x': '' if x is None else round(float(x), 2),
                'y': '' if y is None else round(float(y), 2),
                'status': statuses[idx] if statuses and idx < len(statuses) else '',
                'score': '' if scores is None or idx >= len(scores) else round(float(scores[idx]), 4),
            })


def write_bounce_csv(path, bounces):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                'frame_index', 'time_sec', 'x', 'y', 'score',
                'vy_before', 'vy_after', 'prominence', 'x_velocity_change',
            ],
        )
        writer.writeheader()
        for row in bounces:
            writer.writerow({
                'frame_index': int(row['frame_index']),
                'time_sec': round(float(row['time_sec']), 4),
                'x': round(float(row['x']), 2),
                'y': round(float(row['y']), 2),
                'score': round(float(row['score']), 4),
                'vy_before': round(float(row['vy_before']), 4),
                'vy_after': round(float(row['vy_after']), 4),
                'prominence': round(float(row['prominence']), 4),
                'x_velocity_change': round(float(row['x_velocity_change']), 4),
            })


def write_video(frames, track, bounces, output_path, fps):
    if not frames:
        return
    height, width = frames[0].shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'DIVX'),
        fps,
        (width, height),
    )
    bounce_by_frame = {int(row['frame_index']): row for row in bounces}

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
            x = int(visible['x'])
            y = int(visible['y'])
            cv2.circle(canvas, (x, y), 18, (0, 255, 255), 3)
            cv2.putText(
                canvas,
                'Y-BOUNCE',
                (max(0, x - 45), max(24, y - 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        writer.write(canvas)
    writer.release()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=Path, required=True)
    parser.add_argument('--model_path', type=Path, default=Path('weights/tracknet_pretrained.pt'))
    parser.add_argument('--output_root', type=Path, default=Path('output_y_reversal_bounces'))
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])

    parser.add_argument('--min_confidence', type=float, default=82)
    parser.add_argument('--max_dist', type=float, default=190)
    parser.add_argument('--max_prediction_gap', type=int, default=18)
    parser.add_argument('--max_gate_dist', type=float, default=480)
    parser.add_argument('--min_gate_detections', type=int, default=4)
    parser.add_argument('--offline_window', type=int, default=7)
    parser.add_argument('--offline_polyorder', type=int, default=2)
    parser.add_argument('--centroid_radius', type=int, default=9)
    parser.add_argument('--relative_threshold', type=float, default=0.55)
    parser.add_argument('--max_candidates', type=int, default=5)
    parser.add_argument('--max_candidate_dist', type=float, default=340)
    parser.add_argument('--score_distance_tradeoff', type=float, default=0.35)
    parser.add_argument('--no_visual_refine', action='store_true')
    parser.add_argument('--visual_roi_radius', type=int, default=36)
    parser.add_argument('--no_fast_ball', action='store_true')
    parser.add_argument('--fast_speed_threshold', type=float, default=50)
    parser.add_argument('--fast_min_confidence', type=float, default=55)
    parser.add_argument('--max_bridge_gap', type=int, default=6)
    parser.add_argument('--suppress_isolated', action='store_true', default=True)
    parser.add_argument('--isolation_window', type=int, default=2)
    parser.add_argument('--isolation_max_dist', type=float, default=220)
    parser.add_argument('--suppress_jumps', action='store_true')
    parser.add_argument('--jump_window', type=int, default=6)
    parser.add_argument('--jump_max_interp_error', type=float, default=90)
    parser.add_argument('--jump_max_step_dist', type=float, default=190)

    parser.add_argument('--window', type=int, default=3)
    parser.add_argument('--local_window', type=int, default=4)
    parser.add_argument('--min_local_points', type=int, default=5)
    parser.add_argument('--min_down_speed', type=float, default=3.0)
    parser.add_argument('--min_up_speed', type=float, default=2.0)
    parser.add_argument('--local_y_tolerance', type=float, default=10.0)
    parser.add_argument('--max_x_velocity_change', type=float, default=85.0)
    parser.add_argument('--min_score', type=float, default=9.0)
    parser.add_argument('--min_gap', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device(args.device)
    model = load_model(args.model_path, device)
    frames, fps = read_video(str(args.video_path))
    if not frames:
        raise RuntimeError(f'video has no readable frames: {args.video_path}')

    track, statuses, scores, _, _ = track_ball(frames, model, args)
    frame_height, frame_width = frames[0].shape[:2]
    video_track = scale_track_to_frame(track, frame_width, frame_height)

    output_stem = args.video_path.stem
    track_csv = args.output_root / f'{output_stem}_track.csv'
    bounce_csv = args.output_root / f'{output_stem}_y_bounces.csv'
    video_path = args.output_root / f'{output_stem}_y_bounces.avi'

    write_track_csv(track_csv, video_track, statuses, scores, fps)
    bounces = detect_y_reversal_bounces(video_track, fps, args)
    write_bounce_csv(bounce_csv, bounces)
    write_video(frames, video_track, bounces, video_path, fps)

    print(f'track_csv={track_csv}')
    print(f'bounce_csv={bounce_csv}')
    print(f'video={video_path}')
    print(f'predicted_bounces={len(bounces)}')
    for row in bounces:
        print(
            f"frame={row['frame_index']}, time_sec={row['time_sec']:.3f}, "
            f"x={row['x']:.2f}, y={row['y']:.2f}, score={row['score']:.3f}"
        )


if __name__ == '__main__':
    main()
