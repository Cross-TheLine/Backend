import csv
import json
from types import SimpleNamespace

import cv2
import numpy as np

from src.inout_judgement.judge_in_out import (
    as_point,
    frame_size_from_record,
    line_endpoints,
    normalize_config,
    scale_point,
    select_view2_record,
    video_size,
)


DECISION_COLORS = {
    'IN': (0, 210, 0),
    'OUT': (0, 0, 255),
    'UNKNOWN': (0, 220, 255),
}

DEFAULT_LINE_THICKNESS = 3


def read_csv(path):
    with path.open('r', newline='', encoding='utf-8-sig') as csv_file:
        return list(csv.DictReader(csv_file))


def load_json(path):
    with path.open('r', encoding='utf-8') as json_file:
        return json.load(json_file)


def rounded_point(point):
    return tuple(np.round(np.array(point, dtype=np.float32)).astype(int))


def draw_text_with_backdrop(image, text, origin, font_scale, color, thickness=2):
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = max(6, int(height * 0.35))
    x1 = max(0, x - pad)
    y1 = max(0, y - height - pad)
    x2 = min(image.shape[1] - 1, x + width + pad)
    y2 = min(image.shape[0] - 1, y + baseline + pad)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def select_record_for_video(config_data, args, width, height):
    if isinstance(config_data, dict) and config_data.get('mode') in {'polygon', 'line'}:
        return None
    return select_view2_record(
        config_data,
        config_image=args.config_image,
        config_index=args.config_index,
        target_width=width,
        target_height=height,
    )


def scaled_line_segments(record, target_width, target_height):
    if record is None:
        return []

    source_width, source_height = frame_size_from_record(record)
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    segments = []
    for line in record.get('lines', []):
        points = line_endpoints(line)
        segments.append([
            scale_point(points[0], scale_x, scale_y),
            scale_point(points[1], scale_x, scale_y),
        ])
    return segments


def configured_line_segments(config):
    segments = config.get('view2_line_segments')
    if not segments:
        return []
    return [
        [as_point(segment[0]), as_point(segment[1])]
        for segment in segments
        if len(segment) == 2
    ]


def config_for_video(config_data, args, width, height):
    config_args = SimpleNamespace(
        config_image=args.config_image,
        config_index=args.config_index,
        target_width=width,
        target_height=height,
        video_path=None,
    )
    return normalize_config(config_data, config_args)


def draw_court_overlay(frame, polygon, line_segments, line_thickness):
    if polygon:
        poly = np.array([rounded_point(point) for point in polygon], dtype=np.int32)
        fill = frame.copy()
        cv2.fillPoly(fill, [poly], (0, 120, 0))
        cv2.addWeighted(fill, 0.12, frame, 0.88, 0, frame)
        cv2.polylines(frame, [poly], True, (0, 180, 0), max(1, line_thickness // 2), cv2.LINE_AA)

    for start, end in line_segments:
        cv2.line(frame, rounded_point(start), rounded_point(end), (255, 0, 0), line_thickness, cv2.LINE_AA)


def numeric_value(row, names):
    for name in names:
        value = row.get(name)
        if value not in (None, ''):
            return float(value)
    raise KeyError(f'Missing numeric column from {names}')


def bounce_point(row):
    return (
        numeric_value(row, ['judge_x', 'contact_x', 'x']),
        numeric_value(row, ['judge_y', 'contact_y', 'y']),
    )


def load_bounces(path):
    bounces = []
    for row in read_csv(path):
        if not row:
            continue
        if not row.get('frame_index'):
            continue
        try:
            x, y = bounce_point(row)
        except (KeyError, ValueError):
            continue
        bounces.append({
            'frame_index': int(float(row['frame_index'])),
            'time_sec': row.get('time_sec', ''),
            'x': x,
            'y': y,
            'decision': row.get('decision', 'UNKNOWN') or 'UNKNOWN',
            'reason': row.get('decision_reason', ''),
            'distance': row.get('boundary_distance_px', ''),
        })
    return bounces


def load_track(path):
    if path is None:
        return {}

    track = {}
    for row in read_csv(path):
        if not row.get('frame_index') or not row.get('x') or not row.get('y'):
            continue
        try:
            track[int(float(row['frame_index']))] = {
                'x': float(row['x']),
                'y': float(row['y']),
                'status': row.get('status', ''),
            }
        except ValueError:
            continue
    return track


def draw_track(frame, track, frame_index, trace):
    for offset in range(trace):
        idx = frame_index - offset
        point = track.get(idx)
        if point is None:
            continue
        alpha = 1.0 - offset / max(trace, 1)
        color = (0, int(120 + 120 * alpha), 255)
        radius = max(2, int(5 - offset * 0.28))
        cv2.circle(frame, (int(point['x']), int(point['y'])), radius, color, -1, cv2.LINE_AA)


def visible_bounce(bounces_by_frame, frame_index, display_window):
    for delta in range(display_window + 1):
        for candidate in (frame_index - delta, frame_index + delta):
            bounce = bounces_by_frame.get(candidate)
            if bounce is not None:
                return bounce
    return None


def draw_bounce(frame, bounce):
    x = int(round(bounce['x']))
    y = int(round(bounce['y']))
    decision = bounce['decision'].upper()
    color = DECISION_COLORS.get(decision, DECISION_COLORS['UNKNOWN'])
    label = f'{decision} BOUNCE'

    cv2.circle(frame, (x, y), 22, color, 2, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
    draw_text_with_backdrop(frame, label, (max(12, x - 62), max(34, y - 30)), 0.72, color, 2)


def draw_status_banner(frame, bounce_count, visible):
    if visible is None:
        if bounce_count == 0:
            draw_text_with_backdrop(frame, 'NO BOUNCE', (24, 44), 0.85, (0, 220, 255), 2)
        return

    decision = visible['decision'].upper()
    color = DECISION_COLORS.get(decision, DECISION_COLORS['UNKNOWN'])
    distance = visible.get('distance') or ''
    suffix = f'  dist={distance}px' if distance != '' else ''
    text = f'{decision}  frame={visible["frame_index"]}{suffix}'
    draw_text_with_backdrop(frame, text, (24, 44), 0.85, color, 2)


def write_overlay_video(args):
    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {args.video_path}')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    config_data = load_json(args.court_config)
    record = select_record_for_video(config_data, args, width, height)
    config = config_for_video(config_data, args, width, height)
    polygon = [as_point(point) for point in config['court_polygon']] if config.get('mode') == 'polygon' else []
    line_segments = configured_line_segments(config) or scaled_line_segments(record, width, height)
    if not line_segments and config.get('mode') == 'line':
        line_segments = [[as_point(config['line_start']), as_point(config['line_end'])]]

    bounces = load_bounces(args.judged_csv)
    bounces_by_frame = {bounce['frame_index']: bounce for bounce in bounces}
    track = {} if args.hide_track else load_track(args.track_csv)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output_path),
        cv2.VideoWriter_fourcc(*'DIVX'),
        fps,
        (width, height),
    )

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        draw_court_overlay(frame, polygon, line_segments, args.line_thickness)
        if track:
            draw_track(frame, track, frame_index, args.trace)
        visible = visible_bounce(bounces_by_frame, frame_index, args.bounce_display_window)
        if visible is not None:
            draw_bounce(frame, visible)
        draw_status_banner(frame, len(bounces), visible)

        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()
    return frame_index, len(bounces), width, height, fps
