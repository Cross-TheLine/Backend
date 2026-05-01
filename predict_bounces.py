import argparse
import os

import catboost as ctb
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from general import postprocess
from infer_on_video import interpolation, read_video, remove_outliers, smooth_track, split_track
from model import BallTrackerNet


TRAIN_WIDTH = 1280
TRAIN_HEIGHT = 720
MODEL_WIDTH = 640
MODEL_HEIGHT = 360


def select_device(device_arg):
    if device_arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_arg


def load_tracknet(model_path, device):
    model = BallTrackerNet()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def infer_ball_track(frames, model, device):
    train_track = [(None, None)] * 2
    dists = [-1] * 2

    for idx in tqdm(range(2, len(frames)), desc='TrackNet'):
        img = cv2.resize(frames[idx], (MODEL_WIDTH, MODEL_HEIGHT))
        img_prev = cv2.resize(frames[idx - 1], (MODEL_WIDTH, MODEL_HEIGHT))
        img_preprev = cv2.resize(frames[idx - 2], (MODEL_WIDTH, MODEL_HEIGHT))

        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)

        with torch.no_grad():
            out = model(torch.from_numpy(inp).float().to(device))
        output = out.argmax(dim=1).detach().cpu().numpy()
        x_pred, y_pred = postprocess(output)
        train_track.append((x_pred, y_pred))

        if train_track[-1][0] and train_track[-2][0]:
            dist = np.linalg.norm(np.array(train_track[-1]) - np.array(train_track[-2]))
        else:
            dist = -1
        dists.append(dist)

    train_track = remove_outliers(train_track, dists)
    subtracks = split_track(train_track)
    for start, end in subtracks:
        train_track[start:end] = interpolation(train_track[start:end])
    train_track = smooth_track(train_track)
    return train_track


def to_video_track(train_track, frame_width, frame_height):
    scale_x = frame_width / TRAIN_WIDTH
    scale_y = frame_height / TRAIN_HEIGHT
    video_track = []
    for x, y in train_track:
        if x is None or y is None:
            video_track.append((None, None))
        else:
            video_track.append((x * scale_x, y * scale_y))
    return video_track


def build_feature_table(train_track, num_frames):
    rows = []
    for idx, (x, y) in enumerate(train_track):
        if x is None or y is None:
            continue

        row = {'frame_index': idx, 'x': x, 'y': y}
        valid = True
        eps = 1e-15
        for offset in range(1, num_frames):
            prev_idx = idx - offset
            next_idx = idx + offset
            if prev_idx < 0 or next_idx >= len(train_track):
                valid = False
                break

            x_prev, y_prev = train_track[prev_idx]
            x_next, y_next = train_track[next_idx]
            if None in (x_prev, y_prev, x_next, y_next):
                valid = False
                break

            x_diff = abs(x_prev - x)
            y_diff = y_prev - y
            x_diff_inv = abs(x_next - x)
            y_diff_inv = y_next - y

            row[f'x_diff_{offset}'] = x_diff
            row[f'y_diff_{offset}'] = y_diff
            row[f'x_diff_inv_{offset}'] = x_diff_inv
            row[f'y_diff_inv_{offset}'] = y_diff_inv
            row[f'x_div_{offset}'] = abs(x_diff / (x_diff_inv + eps))
            row[f'y_div_{offset}'] = y_diff / (y_diff_inv + eps)

        if valid:
            rows.append(row)
    return pd.DataFrame(rows)


def get_feature_columns(num_frames):
    colnames_x = [f'x_diff_{i}' for i in range(1, num_frames)]
    colnames_x += [f'x_diff_inv_{i}' for i in range(1, num_frames)]
    colnames_x += [f'x_div_{i}' for i in range(1, num_frames)]
    colnames_y = [f'y_diff_{i}' for i in range(1, num_frames)]
    colnames_y += [f'y_diff_inv_{i}' for i in range(1, num_frames)]
    colnames_y += [f'y_div_{i}' for i in range(1, num_frames)]
    return colnames_x + colnames_y


def predict_bounce_frames(train_track, bounce_model_path, num_frames, threshold, min_gap):
    features_df = build_feature_table(train_track, num_frames)
    if features_df.empty:
        return features_df, []

    model = ctb.CatBoostRegressor()
    model.load_model(bounce_model_path)
    feature_columns = get_feature_columns(num_frames)
    scores = model.predict(features_df[feature_columns])
    features_df = features_df.copy()
    features_df['bounce_score'] = scores

    candidates = features_df[features_df['bounce_score'] >= threshold].copy()
    candidates = candidates.sort_values('bounce_score', ascending=False)

    selected_frames = []
    for _, row in candidates.iterrows():
        frame_index = int(row['frame_index'])
        if any(abs(frame_index - prev) < min_gap for prev in selected_frames):
            continue
        selected_frames.append(frame_index)

    selected_frames.sort()
    return features_df, selected_frames


def write_bounce_csv(path_output_csv, selected_frames, video_track, fps, features_df):
    rows = []
    score_by_frame = {
        int(row['frame_index']): float(row['bounce_score'])
        for _, row in features_df.iterrows()
    }
    for frame_index in selected_frames:
        x, y = video_track[frame_index]
        rows.append({
            'frame_index': frame_index,
            'time_sec': frame_index / fps,
            'x': None if x is None else round(float(x), 2),
            'y': None if y is None else round(float(y), 2),
            'bounce_score': round(score_by_frame.get(frame_index, 0.0), 6),
        })
    pd.DataFrame(rows).to_csv(path_output_csv, index=False)


def write_bounce_video(frames, video_track, selected_frames, path_output_video, fps):
    height, width = frames[0].shape[:2]
    selected_set = set(selected_frames)
    out = cv2.VideoWriter(
        path_output_video,
        cv2.VideoWriter_fourcc(*'DIVX'),
        fps,
        (width, height),
    )

    for idx, frame in enumerate(frames):
        canvas = frame.copy()
        x, y = video_track[idx]
        if x is not None and y is not None:
            canvas = cv2.circle(canvas, (int(x), int(y)), 6, (0, 0, 255), -1)
        if idx in selected_set and x is not None and y is not None:
            canvas = cv2.circle(canvas, (int(x), int(y)), 16, (0, 255, 255), 3)
            canvas = cv2.putText(
                canvas,
                'BOUNCE',
                (max(0, int(x) - 25), max(20, int(y) - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        out.write(canvas)
    out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='path to TrackNet weights')
    parser.add_argument('--bounce_model_path', type=str, required=True, help='path to bounce model (.cbm)')
    parser.add_argument('--video_path', type=str, required=True, help='path to input video')
    parser.add_argument('--csv_out_path', type=str, default='bounce_predictions.csv', help='path to CSV output')
    parser.add_argument('--video_out_path', type=str, default='bounce_predictions.avi', help='path to output video')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='device to use for inference')
    parser.add_argument('--num_feature_frames', type=int, default=3, help='feature window for bounce model')
    parser.add_argument('--threshold', type=float, default=0.5, help='bounce score threshold')
    parser.add_argument('--min_gap', type=int, default=8, help='minimum frame gap between bounce events')
    args = parser.parse_args()

    device = select_device(args.device)
    model = load_tracknet(args.model_path, device)
    frames, fps = read_video(args.video_path)
    if not frames:
        raise ValueError(f'Could not read frames from {args.video_path}')

    train_track = infer_ball_track(frames, model, device)
    frame_height, frame_width = frames[0].shape[:2]
    video_track = to_video_track(train_track, frame_width, frame_height)
    features_df, selected_frames = predict_bounce_frames(
        train_track,
        args.bounce_model_path,
        args.num_feature_frames,
        args.threshold,
        args.min_gap,
    )

    os.makedirs(os.path.dirname(args.csv_out_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.video_out_path) or '.', exist_ok=True)
    write_bounce_csv(args.csv_out_path, selected_frames, video_track, fps, features_df)
    write_bounce_video(frames, video_track, selected_frames, args.video_out_path, fps)

    print(f'predicted_bounces = {len(selected_frames)}')
    for frame_index in selected_frames:
        x, y = video_track[frame_index]
        print(f'frame={frame_index}, time_sec={frame_index / fps:.3f}, x={x}, y={y}')
