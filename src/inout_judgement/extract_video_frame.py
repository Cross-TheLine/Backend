import argparse
from pathlib import Path

import cv2


def frame_label(frame_index, time_sec):
    if time_sec is not None:
        return f'{int(round(time_sec * 1000))}ms'
    return f'frame{frame_index}'


def output_path_for(video_path, output_dir, label, used_paths):
    output_path = output_dir / f'{video_path.stem}_{label}.png'
    if output_path not in used_paths:
        return output_path

    parent = video_path.parent.name
    if parent:
        output_path = output_dir / f'{parent}_{video_path.stem}_{label}.png'
    counter = 2
    while output_path in used_paths:
        output_path = output_dir / f'{video_path.stem}_{label}_{counter}.png'
        counter += 1
    return output_path


def extract_frame(video_path, output_path, frame_index, time_sec):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {video_path}')

    if time_sec is not None:
        if time_sec < 0:
            raise ValueError('--time_sec must be >= 0')
        cap.set(cv2.CAP_PROP_POS_MSEC, float(time_sec) * 1000.0)
    else:
        if frame_index < 0:
            raise ValueError('--frame_index must be >= 0')
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))

    ok, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    actual_frame = int(max(0, round(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)))
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f'Could not read frame from video: {video_path}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f'Could not write frame image: {output_path}')

    height, width = frame.shape[:2]
    return {
        'video': str(video_path),
        'frame': actual_frame,
        'time_sec': round(actual_frame / fps, 4) if fps else '',
        'width': width,
        'height': height,
        'output': str(output_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract one still frame from each video for line detection input.'
    )
    parser.add_argument('--videos', nargs='+', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--frame-index', type=int, default=0)
    parser.add_argument('--time-sec', type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    label = frame_label(args.frame_index, args.time_sec)
    used_paths = set()
    for video_path in args.videos:
        output_path = output_path_for(video_path, args.out_dir, label, used_paths)
        used_paths.add(output_path)
        row = extract_frame(video_path, output_path, args.frame_index, args.time_sec)
        print(
            'video={} frame={} time_sec={} size={}x{} output={}'.format(
                row['video'],
                row['frame'],
                row['time_sec'],
                row['width'],
                row['height'],
                row['output'],
            )
        )


if __name__ == '__main__':
    main()
