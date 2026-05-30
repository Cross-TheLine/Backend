import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from general import postprocess
from infer_on_video import (
    interpolation,
    read_video,
    remove_outliers,
    smooth_track,
    split_track,
)
from model import BallTrackerNet


VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}
TRAIN_WIDTH = 1280
TRAIN_HEIGHT = 720
MODEL_WIDTH = 640
MODEL_HEIGHT = 360


def select_device(device_arg):
    if device_arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_arg


def iter_seg_videos(input_root):
    for path in sorted(input_root.rglob('*')):
        if path.is_file() and 'seg' in path.name.lower() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def output_path_for(video_path, input_root, output_root):
    relative = video_path.relative_to(input_root)
    output_relative = relative.with_name(f'{relative.stem}_track.mp4')
    return output_root / output_relative


def write_track_mp4(frames, ball_track, path_output_video, fps, trace=7):
    height, width = frames[0].shape[:2]
    scale_x = width / TRAIN_WIDTH
    scale_y = height / TRAIN_HEIGHT
    out = cv2.VideoWriter(
        path_output_video,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )
    for num, frame in enumerate(frames):
        canvas = frame.copy()
        for i in range(trace):
            if num - i <= 0:
                continue
            if ball_track[num - i][0]:
                x = int(ball_track[num - i][0] * scale_x)
                y = int(ball_track[num - i][1] * scale_y)
                canvas = cv2.circle(canvas, (x, y), radius=0, color=(0, 0, 255), thickness=10 - i)
            else:
                break
        out.write(canvas)
    out.release()


def remove_motion_outliers(ball_track, max_speed=95, max_accel=80):
    filtered = list(ball_track)
    last_valid_idx = None
    last_valid_point = None
    last_velocity = None

    for idx, point in enumerate(ball_track):
        x, y = point
        if x is None or y is None:
            continue

        current = np.array([x, y], dtype=np.float32)
        if last_valid_point is None:
            last_valid_idx = idx
            last_valid_point = current
            continue

        frame_gap = max(1, idx - last_valid_idx)
        velocity = (current - last_valid_point) / frame_gap
        speed = float(np.linalg.norm(velocity))

        if speed > max_speed:
            filtered[idx] = (None, None)
            continue

        if last_velocity is not None:
            accel = float(np.linalg.norm(velocity - last_velocity))
            if accel > max_accel:
                filtered[idx] = (None, None)
                continue

        last_valid_idx = idx
        last_valid_point = current
        last_velocity = velocity

    return filtered


def kalman_smooth_track(ball_track):
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 1]],
        dtype=np.float32,
    )
    kalman.measurementMatrix = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0]],
        dtype=np.float32,
    )
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.5
    kalman.errorCovPost = np.eye(4, dtype=np.float32)

    smoothed = []
    initialized = False
    missing_count = 0

    for x, y in ball_track:
        if x is None or y is None:
            if initialized and missing_count <= 3:
                prediction = kalman.predict()
                smoothed.append((float(prediction[0]), float(prediction[1])))
                missing_count += 1
            else:
                smoothed.append((None, None))
            continue

        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        if not initialized:
            kalman.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
            initialized = True
            smoothed.append((float(x), float(y)))
        else:
            kalman.predict()
            corrected = kalman.correct(measurement)
            smoothed.append((float(corrected[0]), float(corrected[1])))
        missing_count = 0

    return smoothed


def infer_model_batched(frames, model, device, batch_size):
    dists = [-1] * 2
    ball_track = [(None, None)] * 2
    resized = [cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT)) for frame in frames]

    with torch.no_grad():
        for start in tqdm(range(2, len(frames), batch_size)):
            end = min(len(frames), start + batch_size)
            batch_inputs = []
            for idx in range(start, end):
                imgs = np.concatenate((resized[idx], resized[idx - 1], resized[idx - 2]), axis=2)
                imgs = imgs.astype(np.float32) / 255.0
                imgs = np.rollaxis(imgs, 2, 0)
                batch_inputs.append(imgs)

            inp = np.stack(batch_inputs, axis=0)
            out = model(torch.from_numpy(inp).float().to(device))
            outputs = out.argmax(dim=1).detach().cpu().numpy()

            for output in outputs:
                x_pred, y_pred = postprocess(output)
                ball_track.append((x_pred, y_pred))

                if ball_track[-1][0] and ball_track[-2][0]:
                    dist = np.linalg.norm(np.array(ball_track[-1]) - np.array(ball_track[-2]))
                else:
                    dist = -1
                dists.append(dist)

    return ball_track, dists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--input_root', required=True)
    parser.add_argument('--output_root', default='output')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--match', default='')
    parser.add_argument('--skip_existing', action='store_true')
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    model = BallTrackerNet()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    videos = list(iter_seg_videos(input_root))
    if args.match:
        videos = [video for video in videos if args.match.lower() in video.name.lower()]
    if args.limit > 0:
        videos = videos[:args.limit]
    print(f'found_seg_videos = {len(videos)}')
    print(f'device = {device}')

    for index, video_path in enumerate(videos, start=1):
        out_path = output_path_for(video_path, input_root, output_root)
        if args.skip_existing and out_path.exists():
            print(f'[{index}/{len(videos)}] skip existing: {out_path}')
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'[{index}/{len(videos)}] input = {video_path}')
        print(f'[{index}/{len(videos)}] output = {out_path}')

        frames, fps = read_video(str(video_path))
        if not frames:
            print(f'[{index}/{len(videos)}] no frames, skipped')
            continue

        ball_track, dists = infer_model_batched(frames, model, device, args.batch_size)
        ball_track = remove_outliers(ball_track, dists)
        ball_track = remove_motion_outliers(ball_track)
        for start, end in split_track(ball_track):
            ball_track[start:end] = interpolation(ball_track[start:end])
        ball_track = smooth_track(ball_track)
        ball_track = kalman_smooth_track(ball_track)
        write_track_mp4(frames, ball_track, str(out_path), fps)


if __name__ == '__main__':
    main()
