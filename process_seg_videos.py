import argparse
import csv
from collections import Counter
from pathlib import Path

import torch

import infer_on_video as infer_module
from infer_on_video import (
    MIN_TRACK_CONFIDENCE,
    infer_model,
    kalman_smooth_track,
    offline_smooth_track,
    read_video,
    remove_outliers,
    smooth_track,
    suppress_isolated_detections,
    suppress_jump_detections,
    write_track,
    bridge_short_gaps,
)
from model import BallTrackerNet

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


def output_path_for(video_path, input_root, output_root, suffix):
    rel_path = video_path.relative_to(input_root)
    return output_root / rel_path.parent / f'{video_path.stem}{suffix}.avi'


def process_video(video_path, input_root, output_root, debug_root, model, args):
    frames, fps = read_video(str(video_path))
    if not frames:
        raise RuntimeError('video has no readable frames')

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

    output_path = output_path_for(video_path, input_root, output_root, '_ball_detected')
    debug_path = output_path_for(video_path, input_root, debug_root, '_debug')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    write_track(
        frames,
        ball_track,
        str(output_path),
        fps,
        statuses=statuses,
        scores=scores,
        detected_only=not args.show_predictions,
    )
    write_track(
        frames,
        ball_track,
        str(debug_path),
        fps,
        statuses=statuses,
        scores=scores,
        debug=True,
    )

    counts = Counter(statuses)
    return {
        'video': str(video_path.relative_to(input_root)),
        'frames': len(frames),
        'raw_detected': raw_detected,
        'after_outlier': after_outlier,
        'detected': counts.get('detected', 0),
        'predicted': counts.get('predicted', 0),
        'rejected': counts.get('rejected', 0),
        'isolated': counts.get('isolated', 0),
        'jump': counts.get('jump', 0),
        'missing': counts.get('missing', 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', type=Path, default=Path('테니스 데이터'))
    parser.add_argument('--output_root', type=Path, default=Path('output'))
    parser.add_argument('--debug_root', type=Path, default=Path('output_debug'))
    parser.add_argument('--model_path', type=Path, default=Path('weights/tracknet_pretrained.pt'))
    parser.add_argument('--pattern', type=str, default='*seg*.mp4')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--min_confidence', type=float, default=MIN_TRACK_CONFIDENCE)
    parser.add_argument('--max_prediction_gap', type=int, default=16)
    parser.add_argument('--max_dist', type=float, default=140)
    parser.add_argument('--max_gate_dist', type=float, default=360)
    parser.add_argument('--min_gate_detections', type=int, default=4)
    parser.add_argument('--offline_window', type=int, default=7)
    parser.add_argument('--offline_polyorder', type=int, default=2)
    parser.add_argument('--centroid_radius', type=int, default=8)
    parser.add_argument('--relative_threshold', type=float, default=0.6)
    parser.add_argument('--max_candidates', type=int, default=5)
    parser.add_argument('--max_candidate_dist', type=float, default=260)
    parser.add_argument('--score_distance_tradeoff', type=float, default=0.35)
    parser.add_argument('--no_visual_refine', action='store_true')
    parser.add_argument('--visual_roi_radius', type=int, default=30)
    parser.add_argument('--no_fast_ball', action='store_true')
    parser.add_argument('--fast_speed_threshold', type=float, default=70)
    parser.add_argument('--fast_min_confidence', type=float, default=55)
    parser.add_argument('--max_bridge_gap', type=int, default=8)
    parser.add_argument('--show_predictions', action='store_true')
    parser.add_argument('--suppress_isolated', action='store_true')
    parser.add_argument('--isolation_window', type=int, default=3)
    parser.add_argument('--isolation_max_dist', type=float, default=140)
    parser.add_argument('--suppress_jumps', action='store_true')
    parser.add_argument('--jump_window', type=int, default=6)
    parser.add_argument('--jump_max_interp_error', type=float, default=90)
    parser.add_argument('--jump_max_step_dist', type=float, default=190)
    args = parser.parse_args()

    device = select_device(args.device)
    model = load_model(args.model_path, device)
    videos = sorted(args.input_root.rglob(args.pattern))
    rows = []
    errors = []

    for idx, video_path in enumerate(videos, start=1):
        print(f'[{idx}/{len(videos)}] {video_path}')
        try:
            rows.append(process_video(video_path, args.input_root, args.output_root, args.debug_root, model, args))
        except Exception as exc:
            errors.append({'video': str(video_path.relative_to(args.input_root)), 'error': str(exc)})

    summary_path = args.output_root / 'seg_detection_summary.csv'
    debug_summary_path = args.debug_root / 'tracking_status_summary.csv'
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.debug_root.mkdir(parents=True, exist_ok=True)

    fields = [
        'video', 'frames', 'raw_detected', 'after_outlier', 'detected',
        'predicted', 'rejected', 'isolated', 'jump', 'missing',
    ]
    for path in (summary_path, debug_summary_path):
        with path.open('w', newline='', encoding='utf-8-sig') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    if errors:
        error_path = args.debug_root / 'processing_errors.csv'
        with error_path.open('w', newline='', encoding='utf-8-sig') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['video', 'error'])
            writer.writeheader()
            writer.writerows(errors)

    print('processed={}, errors={}'.format(len(rows), len(errors)))
    print('summary={}'.format(summary_path))
    print('debug_summary={}'.format(debug_summary_path))


if __name__ == '__main__':
    main()
