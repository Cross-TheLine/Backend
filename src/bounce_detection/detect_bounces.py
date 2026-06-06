import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

import src.ball_tracking.infer_on_video as infer_module
from src.ball_tracking.infer_on_video import (
    MIN_TRACK_CONFIDENCE,
    TRACK_HEIGHT,
    TRACK_WIDTH,
    bridge_short_gaps,
    infer_model,
    kalman_smooth_track,
    offline_smooth_track,
    read_video,
    remove_outliers,
    smooth_track,
    suppress_isolated_detections,
    suppress_jump_detections,
)
from src.ball_tracking.model import BallTrackerNet


infer_module.tqdm = lambda iterable, *args, **kwargs: iterable


def select_device(device_arg):
    if device_arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_arg


def load_model(model_path, device):
    model = BallTrackerNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def scale_track_to_frame(track, frame_width, frame_height):
    scale_x = frame_width / TRACK_WIDTH
    scale_y = frame_height / TRACK_HEIGHT
    scaled = []
    for x, y in track:
        if x is None or y is None:
            scaled.append((None, None))
        else:
            scaled.append((float(x) * scale_x, float(y) * scale_y))
    return scaled


def track_ball(frames, model, args):
    ball_track, dists, scores = infer_model(
        frames,
        model,
        return_scores=True,
        min_confidence=args.min_confidence,
        centroid_radius=args.centroid_radius,
        relative_threshold=args.relative_threshold,
        max_candidates=args.max_candidates,
        max_candidate_dist=args.max_candidate_dist,
        score_distance_tradeoff=args.score_distance_tradeoff,
        visual_refine=not args.no_visual_refine,
        visual_roi_radius=args.visual_roi_radius,
        fast_ball=not args.no_fast_ball,
        fast_speed_threshold=args.fast_speed_threshold,
        fast_min_confidence=args.fast_min_confidence,
    )
    raw_detected = sum(1 for x, y in ball_track if x is not None and y is not None)
    ball_track = remove_outliers(ball_track, dists, max_dist=args.max_dist)
    after_outlier = sum(1 for x, y in ball_track if x is not None and y is not None)

    ball_track, statuses = kalman_smooth_track(
        ball_track,
        scores,
        min_confidence=args.min_confidence,
        max_prediction_gap=args.max_prediction_gap,
        max_gate_dist=args.max_gate_dist,
        min_gate_detections=args.min_gate_detections,
        return_statuses=True,
    )
    ball_track, statuses = bridge_short_gaps(
        ball_track,
        statuses,
        max_gap=args.max_bridge_gap,
    )
    ball_track = offline_smooth_track(
        ball_track,
        window_size=args.offline_window,
        polyorder=args.offline_polyorder,
    )
    ball_track = smooth_track(ball_track)

    if args.suppress_isolated:
        statuses = suppress_isolated_detections(
            ball_track,
            statuses,
            window=args.isolation_window,
            max_link_dist=args.isolation_max_dist,
        )
    if args.suppress_jumps:
        statuses = suppress_jump_detections(
            ball_track,
            statuses,
            window=args.jump_window,
            max_interp_error=args.jump_max_interp_error,
            max_step_dist=args.jump_max_step_dist,
        )

    return ball_track, statuses, scores, raw_detected, after_outlier


def point_at(track, idx):
    if idx < 0 or idx >= len(track):
        return None
    x, y = track[idx]
    if x is None or y is None:
        return None
    return float(x), float(y)


def direct_detection_count(statuses, start, end):
    if statuses is None:
        return 0
    start = max(0, start)
    end = min(len(statuses), end)
    return sum(1 for idx in range(start, end) if statuses[idx] == 'detected')


def detect_bounces(track, statuses, fps, args):
    candidates = []
    radius = args.bounce_window
    local_radius = args.local_window
    n = len(track)

    for idx in range(radius, n - radius):
        center = point_at(track, idx)
        before = point_at(track, idx - radius)
        after = point_at(track, idx + radius)
        if center is None or before is None or after is None:
            continue

        x, y = center
        prev_x, prev_y = before
        next_x, next_y = after
        vy_in = (y - prev_y) / radius
        vy_out = (next_y - y) / radius
        vx_in = (x - prev_x) / radius
        vx_out = (next_x - x) / radius

        if vy_in < args.min_down_speed or vy_out > -args.min_up_speed:
            continue

        local_points = [
            point_at(track, j)
            for j in range(idx - local_radius, idx + local_radius + 1)
        ]
        local_points = [p for p in local_points if p is not None]
        if len(local_points) < args.min_local_points:
            continue

        max_y = max(p[1] for p in local_points)
        if y < max_y - args.local_y_tolerance:
            continue

        dx_change = abs(vx_out - vx_in)
        if dx_change > args.max_x_velocity_change:
            continue

        direct_hits = direct_detection_count(
            statuses,
            idx - local_radius,
            idx + local_radius + 1,
        )
        if direct_hits < args.min_direct_detections:
            continue

        reversal = vy_in - vy_out
        prominence = y - min(prev_y, next_y)
        score = reversal + max(prominence, 0.0) * 0.35 - dx_change * 0.12
        candidates.append({
            'frame_index': idx,
            'time_sec': idx / fps if fps else 0.0,
            'x': x,
            'y': y,
            'score': score,
            'vy_in': vy_in,
            'vy_out': vy_out,
            'direct_detections': direct_hits,
        })

    candidates.sort(key=lambda row: row['score'], reverse=True)
    selected = []
    for candidate in candidates:
        if candidate['score'] < args.min_bounce_score:
            continue
        if any(abs(candidate['frame_index'] - row['frame_index']) < args.min_bounce_gap for row in selected):
            continue
        selected.append(candidate)

    selected.sort(key=lambda row: row['frame_index'])
    return selected, candidates


def bounce_contact_offset(frame_height, args):
    if args.bounce_contact_offset >= 0:
        return float(args.bounce_contact_offset)
    return max(4.0, min(18.0, frame_height / 90.0))


def collect_detected_points(track, statuses, start, end):
    points = []
    start = max(0, start)
    end = min(len(track), end)
    for idx in range(start, end):
        if statuses is not None and idx < len(statuses) and statuses[idx] != 'detected':
            continue
        point = point_at(track, idx)
        if point is None:
            continue
        points.append((idx, point[0], point[1]))
    return points


def fit_xy_over_time(points):
    if len(points) < 2:
        return None
    frames = np.array([p[0] for p in points], dtype=np.float32)
    xs = np.array([p[1] for p in points], dtype=np.float32)
    ys = np.array([p[2] for p in points], dtype=np.float32)
    x_coef = np.polyfit(frames, xs, 1)
    y_coef = np.polyfit(frames, ys, 1)
    return x_coef, y_coef


def eval_fit(fit, frame_index):
    x_coef, y_coef = fit
    return float(np.polyval(x_coef, frame_index)), float(np.polyval(y_coef, frame_index))


def estimate_trajectory_contact(row, track, statuses, frame_height, args):
    idx = int(row['frame_index'])
    before = collect_detected_points(
        track,
        statuses,
        idx - args.trajectory_window,
        idx,
    )
    after = collect_detected_points(
        track,
        statuses,
        idx + 1,
        idx + args.trajectory_window + 1,
    )
    if len(before) < args.min_trajectory_points or len(after) < args.min_trajectory_points:
        return None

    before_fit = fit_xy_over_time(before)
    after_fit = fit_xy_over_time(after)
    if before_fit is None or after_fit is None:
        return None

    search_start = max(0.0, idx - args.trajectory_search_radius)
    search_end = min(len(track) - 1.0, idx + args.trajectory_search_radius)
    sample_count = max(9, int((search_end - search_start) * 4) + 1)
    best = None
    for frame_pos in np.linspace(search_start, search_end, sample_count):
        before_x, before_y = eval_fit(before_fit, frame_pos)
        after_x, after_y = eval_fit(after_fit, frame_pos)
        gap = float(np.linalg.norm(np.array([before_x - after_x, before_y - after_y])))
        center_y = (before_y + after_y) * 0.5
        score = gap - center_y * 0.02
        if best is None or score < best[0]:
            best = (score, frame_pos, (before_x + after_x) * 0.5, center_y, gap)

    if best is None or best[4] > args.max_trajectory_gap:
        return None

    _, frame_pos, center_x, center_y, gap = best
    offset = bounce_contact_offset(frame_height, args)
    return {
        'trajectory_frame': float(frame_pos),
        'trajectory_center_x': float(center_x),
        'trajectory_center_y': float(center_y),
        'trajectory_contact_x': float(center_x),
        'trajectory_contact_y': min(float(center_y) + offset, frame_height - 1.0),
        'trajectory_gap': float(gap),
        'trajectory_points_before': len(before),
        'trajectory_points_after': len(after),
    }


def add_contact_points(bounces, track, statuses, frame_height, args):
    offset = bounce_contact_offset(frame_height, args)
    result = []
    for row in bounces:
        corrected = dict(row)
        raw_contact_x = float(row['x'])
        raw_contact_y = min(float(row['y']) + offset, frame_height - 1.0)
        corrected['center_x'] = float(row['x'])
        corrected['center_y'] = float(row['y'])
        corrected['raw_contact_x'] = raw_contact_x
        corrected['raw_contact_y'] = raw_contact_y
        corrected['contact_x'] = raw_contact_x
        corrected['contact_y'] = raw_contact_y
        corrected['contact_offset'] = offset
        corrected['trajectory_used'] = 0
        corrected['trajectory_frame'] = ''
        corrected['trajectory_center_x'] = ''
        corrected['trajectory_center_y'] = ''
        corrected['trajectory_contact_x'] = ''
        corrected['trajectory_contact_y'] = ''
        corrected['trajectory_gap'] = ''
        corrected['trajectory_points_before'] = 0
        corrected['trajectory_points_after'] = 0

        trajectory = estimate_trajectory_contact(row, track, statuses, frame_height, args)
        if trajectory is not None:
            weight = args.trajectory_blend
            corrected.update(trajectory)
            corrected['trajectory_used'] = 1
            corrected['contact_x'] = (
                raw_contact_x * (1.0 - weight) +
                trajectory['trajectory_contact_x'] * weight
            )
            corrected['contact_y'] = (
                raw_contact_y * (1.0 - weight) +
                trajectory['trajectory_contact_y'] * weight
            )
        result.append(corrected)
    return result


def write_bounce_csv(path, bounces):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'frame_index', 'time_sec', 'x', 'y', 'center_x', 'center_y',
        'raw_contact_x', 'raw_contact_y', 'contact_x', 'contact_y',
        'contact_offset', 'trajectory_used', 'trajectory_frame',
        'trajectory_center_x', 'trajectory_center_y', 'trajectory_contact_x',
        'trajectory_contact_y', 'trajectory_gap', 'trajectory_points_before',
        'trajectory_points_after', 'score', 'vy_in', 'vy_out',
        'direct_detections',
    ]
    with path.open('w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in bounces:
            writer.writerow({
                'frame_index': int(row['frame_index']),
                'time_sec': round(float(row['time_sec']), 4),
                'x': round(float(row['contact_x']), 2),
                'y': round(float(row['contact_y']), 2),
                'center_x': round(float(row['center_x']), 2),
                'center_y': round(float(row['center_y']), 2),
                'raw_contact_x': round(float(row['raw_contact_x']), 2),
                'raw_contact_y': round(float(row['raw_contact_y']), 2),
                'contact_x': round(float(row['contact_x']), 2),
                'contact_y': round(float(row['contact_y']), 2),
                'contact_offset': round(float(row['contact_offset']), 2),
                'trajectory_used': int(row['trajectory_used']),
                'trajectory_frame': '' if row['trajectory_frame'] == '' else round(float(row['trajectory_frame']), 3),
                'trajectory_center_x': '' if row['trajectory_center_x'] == '' else round(float(row['trajectory_center_x']), 2),
                'trajectory_center_y': '' if row['trajectory_center_y'] == '' else round(float(row['trajectory_center_y']), 2),
                'trajectory_contact_x': '' if row['trajectory_contact_x'] == '' else round(float(row['trajectory_contact_x']), 2),
                'trajectory_contact_y': '' if row['trajectory_contact_y'] == '' else round(float(row['trajectory_contact_y']), 2),
                'trajectory_gap': '' if row['trajectory_gap'] == '' else round(float(row['trajectory_gap']), 2),
                'trajectory_points_before': int(row['trajectory_points_before']),
                'trajectory_points_after': int(row['trajectory_points_after']),
                'score': round(float(row['score']), 4),
                'vy_in': round(float(row['vy_in']), 4),
                'vy_out': round(float(row['vy_out']), 4),
                'direct_detections': int(row['direct_detections']),
            })


def write_bounce_video(frames, track, statuses, bounces, path, fps, show_track=True,
                       trace=7, bounce_display_window=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    bounce_by_frame = {int(row['frame_index']): row for row in bounces}
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

    for idx, frame in enumerate(frames):
        canvas = frame.copy()
        if show_track:
            for offset in range(trace):
                track_idx = idx - offset
                if track_idx <= 0 or track_idx >= len(track):
                    continue
                x, y = track[track_idx]
                status = statuses[track_idx] if statuses and track_idx < len(statuses) else 'missing'
                if x is None or y is None or status != 'detected':
                    continue
                cv2.circle(
                    canvas,
                    (int(x), int(y)),
                    radius=0,
                    color=(0, 0, 255),
                    thickness=max(1, 10 - offset),
                )

        visible_bounce = None
        if idx in bounce_by_frame:
            visible_bounce = bounce_by_frame[idx]
        else:
            for delta in range(1, bounce_display_window + 1):
                if idx - delta in bounce_by_frame:
                    visible_bounce = bounce_by_frame[idx - delta]
                    break
                if idx + delta in bounce_by_frame:
                    visible_bounce = bounce_by_frame[idx + delta]
                    break

        if visible_bounce is not None:
            row = visible_bounce
            x = int(row.get('contact_x', row['x']))
            y = int(row.get('contact_y', row['y']))
            cv2.circle(canvas, (x, y), 22, (0, 255, 255), 3)
            cv2.circle(canvas, (x, y), 4, (0, 255, 255), -1)
            cv2.putText(
                canvas,
                'BOUNCE',
                (max(0, x - 38), max(24, y - 26)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        out.write(canvas)
    out.release()


def output_stem(video_path, input_root):
    try:
        rel = video_path.relative_to(input_root)
        return rel.with_suffix('')
    except ValueError:
        return Path(video_path.stem)


def process_video(video_path, input_root, output_root, model, args):
    frames, fps = read_video(str(video_path))
    if not frames:
        raise RuntimeError('video has no readable frames')

    track, statuses, scores, raw_detected, after_outlier = track_ball(frames, model, args)
    frame_height, frame_width = frames[0].shape[:2]
    video_track = scale_track_to_frame(track, frame_width, frame_height)
    bounces, candidates = detect_bounces(video_track, statuses, fps, args)
    bounces = add_contact_points(bounces, video_track, statuses, frame_height, args)

    stem = output_stem(video_path, input_root)
    csv_path = output_root / stem.parent / f'{stem.name}_bounces.csv'
    video_path_out = output_root / stem.parent / f'{stem.name}_bounces.avi'
    write_bounce_csv(csv_path, bounces)
    write_bounce_video(
        frames,
        video_track,
        statuses,
        bounces,
        video_path_out,
        fps,
        show_track=not args.hide_track,
        trace=args.trace,
        bounce_display_window=args.bounce_display_window,
    )

    counts = Counter(statuses)
    return {
        'video': str(video_path.relative_to(input_root)),
        'frames': len(frames),
        'raw_detected': raw_detected,
        'after_outlier': after_outlier,
        'detected': counts.get('detected', 0),
        'predicted': counts.get('predicted', 0),
        'isolated': counts.get('isolated', 0),
        'jump': counts.get('jump', 0),
        'missing': counts.get('missing', 0),
        'bounce_count': len(bounces),
        'candidate_count': len(candidates),
        'trajectory_corrected': sum(1 for row in bounces if row.get('trajectory_used') == 1),
        'csv_path': str(csv_path),
        'video_path': str(video_path_out),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', type=Path, default=Path('.'))
    parser.add_argument('--video_path', type=Path)
    parser.add_argument('--pattern', type=str, default='*seg*.mp4')
    parser.add_argument('--output_root', type=Path, default=Path('output_bounces'))
    parser.add_argument('--model_path', type=Path, default=Path('weights/tracknet_pretrained.pt'))
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])

    parser.add_argument('--min_confidence', type=float, default=82)
    parser.add_argument('--max_prediction_gap', type=int, default=18)
    parser.add_argument('--max_dist', type=float, default=190)
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

    parser.add_argument('--bounce_window', type=int, default=3)
    parser.add_argument('--local_window', type=int, default=4)
    parser.add_argument('--min_local_points', type=int, default=5)
    parser.add_argument('--min_down_speed', type=float, default=3.0)
    parser.add_argument('--min_up_speed', type=float, default=2.0)
    parser.add_argument('--local_y_tolerance', type=float, default=10.0)
    parser.add_argument('--max_x_velocity_change', type=float, default=85.0)
    parser.add_argument('--min_direct_detections', type=int, default=2)
    parser.add_argument('--min_bounce_score', type=float, default=9.0)
    parser.add_argument('--min_bounce_gap', type=int, default=10)
    parser.add_argument('--bounce_contact_offset', type=float, default=-1,
                        help='pixels to move bounce marker below ball center; negative uses auto')
    parser.add_argument('--trajectory_window', type=int, default=8)
    parser.add_argument('--min_trajectory_points', type=int, default=2)
    parser.add_argument('--trajectory_search_radius', type=float, default=3.0)
    parser.add_argument('--max_trajectory_gap', type=float, default=130.0)
    parser.add_argument('--trajectory_blend', type=float, default=0.5)
    parser.add_argument('--trace', type=int, default=7)
    parser.add_argument('--bounce_display_window', type=int, default=3)
    parser.add_argument('--hide_track', action='store_true')
    args = parser.parse_args()

    device = select_device(args.device)
    model = load_model(args.model_path, device)

    if args.video_path:
        videos = [args.video_path]
        input_root = args.video_path.parent
    else:
        input_root = args.input_root
        videos = sorted(input_root.rglob(args.pattern))

    rows = []
    errors = []
    for idx, video_path in enumerate(videos, start=1):
        print(f'[{idx}/{len(videos)}] {video_path}')
        try:
            rows.append(process_video(video_path, input_root, args.output_root, model, args))
        except Exception as exc:
            errors.append({'video': str(video_path), 'error': str(exc)})

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / 'bounce_summary.csv'
    fields = [
        'video', 'frames', 'raw_detected', 'after_outlier', 'detected',
        'predicted', 'isolated', 'jump', 'missing', 'bounce_count',
        'candidate_count', 'trajectory_corrected', 'csv_path', 'video_path',
    ]
    with summary_path.open('w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    if errors:
        error_path = args.output_root / 'bounce_errors.csv'
        with error_path.open('w', newline='', encoding='utf-8-sig') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['video', 'error'])
            writer.writeheader()
            writer.writerows(errors)

    print('processed={}, errors={}'.format(len(rows), len(errors)))
    print('summary={}'.format(summary_path))


if __name__ == '__main__':
    main()
