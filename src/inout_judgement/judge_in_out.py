import argparse
import csv
import json
import math
from pathlib import Path


def load_json(path):
    with open(path, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)


def read_csv(path):
    with open(path, 'r', newline='', encoding='utf-8-sig') as csv_file:
        return list(csv.DictReader(csv_file))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_point(value):
    if len(value) != 2:
        raise ValueError(f'Point must have two values: {value}')
    return float(value[0]), float(value[1])


def frame_size_from_record(record):
    if 'width' in record and 'height' in record:
        return float(record['width']), float(record['height'])
    if 'image_shape_hwc' in record:
        height, width = record['image_shape_hwc'][:2]
        return float(width), float(height)
    raise KeyError('View2 line record must include width/height or image_shape_hwc')


def video_size(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video for size detection: {path}')
    width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f'Could not read video size: {path}')
    return width, height


def closest_record_by_size(records, target_width, target_height):
    def score(record):
        width, height = frame_size_from_record(record)
        aspect_error = abs((width / height) - (target_width / target_height)) * 1000.0
        size_error = abs(width - target_width) + abs(height - target_height)
        return aspect_error + size_error

    return min(records, key=score)


def select_view2_record(data, config_image=None, config_index=0,
                        target_width=None, target_height=None):
    records = data if isinstance(data, list) else [data]
    if not records:
        raise ValueError('Court config does not contain any records')

    if config_image:
        wanted = Path(config_image).name
        matches = [
            record for record in records
            if Path(str(record.get('image', ''))).name == wanted
        ]
        if not matches:
            raise ValueError(f'No view2 record found for image: {config_image}')
        return matches[0]

    if target_width is not None and target_height is not None:
        return closest_record_by_size(records, target_width, target_height)

    if len(records) == 1:
        return records[0]

    if config_index < 0 or config_index >= len(records):
        raise IndexError(f'config_index out of range: {config_index}')
    return records[config_index]


def line_endpoints(line):
    points = (
        line.get('points') or
        line.get('extended_endpoints') or
        line.get('segment_endpoints')
    )
    if points is None or len(points) != 2:
        raise ValueError(f'View2 line needs two endpoints: {line}')
    return [as_point(point) for point in points]


def distance_to_frame_border(point, width, height):
    x, y = point
    return min(x, y, width - 1.0 - x, height - 1.0 - y)


def split_border_and_vertex(endpoint_a, endpoint_b, width, height):
    if distance_to_frame_border(endpoint_a, width, height) <= distance_to_frame_border(endpoint_b, width, height):
        return endpoint_a, endpoint_b
    return endpoint_b, endpoint_a


def perimeter_position(point, width, height):
    x, y = point
    right = width - 1.0
    bottom = height - 1.0
    distances = {
        'top': abs(y),
        'right': abs(right - x),
        'bottom': abs(bottom - y),
        'left': abs(x),
    }
    edge = min(distances, key=distances.get)
    if edge == 'top':
        return max(0.0, min(right, x))
    if edge == 'right':
        return right + max(0.0, min(bottom, y))
    if edge == 'bottom':
        return right + bottom + max(0.0, min(right, right - x))
    return right + bottom + right + max(0.0, min(bottom, bottom - y))


def clockwise_boundary_arc(start, end, width, height):
    right = width - 1.0
    bottom = height - 1.0
    perimeter = 2.0 * (right + bottom)
    corner_points = [
        (0.0, 0.0),
        (right, 0.0),
        (right, bottom),
        (0.0, bottom),
    ]
    corner_positions = [0.0, right, right + bottom, right + bottom + right]

    start_pos = perimeter_position(start, width, height)
    end_pos = perimeter_position(end, width, height)
    if end_pos < start_pos:
        end_pos += perimeter

    arc = [start]
    for corner, position in zip(corner_points, corner_positions):
        for shifted in (position, position + perimeter):
            if start_pos < shifted < end_pos:
                arc.append(corner)
    arc.append(end)
    return arc


def top_boundary_arc(start, end, width, height):
    clockwise = clockwise_boundary_arc(start, end, width, height)
    counter_clockwise = list(reversed(clockwise_boundary_arc(end, start, width, height)))

    def average_y(points):
        return sum(point[1] for point in points) / max(len(points), 1)

    if average_y(clockwise) <= average_y(counter_clockwise):
        return clockwise
    return counter_clockwise


def scale_point(point, scale_x, scale_y):
    return float(point[0]) * scale_x, float(point[1]) * scale_y


def view2_lines_to_polygon_config(record, target_width=None, target_height=None):
    width, height = frame_size_from_record(record)
    target_width = width if target_width is None else float(target_width)
    target_height = height if target_height is None else float(target_height)
    scale_x = target_width / width
    scale_y = target_height / height

    lines = record.get('lines', [])
    named_lines = {line.get('name'): line for line in lines}
    top_line = named_lines.get('top_to_vertex')
    side_line = named_lines.get('side_to_vertex')
    if top_line is None or side_line is None:
        if len(lines) != 2:
            raise ValueError('View2 config must contain top_to_vertex and side_to_vertex lines')
        top_line, side_line = lines

    top_border, top_vertex = split_border_and_vertex(
        *line_endpoints(top_line),
        width,
        height,
    )
    side_border, side_vertex = split_border_and_vertex(
        *line_endpoints(side_line),
        width,
        height,
    )

    top_border = scale_point(top_border, scale_x, scale_y)
    top_vertex = scale_point(top_vertex, scale_x, scale_y)
    side_border = scale_point(side_border, scale_x, scale_y)
    side_vertex = scale_point(side_vertex, scale_x, scale_y)

    polygon = top_boundary_arc(top_border, side_border, target_width, target_height)
    polygon.extend([side_vertex, top_vertex])

    return {
        'mode': 'polygon',
        'court_polygon': [[round(x, 3), round(y, 3)] for x, y in polygon],
        'line_tolerance_px': 3.0,
        'source_mode': 'view2_lines',
        'source_image': record.get('image', ''),
        'source_view_side': record.get('view_side') or record.get('layout', ''),
        'source_size': [width, height],
        'target_size': [target_width, target_height],
    }


def normalize_config(data, args):
    if isinstance(data, dict) and data.get('mode') in {'polygon', 'line'}:
        return data

    target_width = args.target_width
    target_height = args.target_height
    if args.video_path:
        target_width, target_height = video_size(args.video_path)

    record = select_view2_record(
        data,
        config_image=args.config_image,
        config_index=args.config_index,
        target_width=target_width,
        target_height=target_height,
    )
    if not (record.get('view') == 'view2' or record.get('mode') == 'view2'):
        raise ValueError('Only polygon/line configs and view2 line records are supported')
    return view2_lines_to_polygon_config(record, target_width, target_height)


def signed_distance_to_line(point, line_start, line_end):
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        raise ValueError('Line points must not be identical')
    return ((px - x1) * dy - (py - y1) * dx) / length


def distance_to_segment(point, seg_start, seg_end):
    px, py = point
    x1, y1 = seg_start
    x2, y2 = seg_end
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / denom))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    count = len(polygon)
    for idx in range(count):
        x1, y1 = polygon[idx]
        x2, y2 = polygon[(idx + 1) % count]
        intersects = (y1 > y) != (y2 > y)
        if intersects:
            x_at_y = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_at_y:
                inside = not inside
    return inside


def distance_to_polygon(point, polygon):
    distances = [
        distance_to_segment(point, polygon[idx], polygon[(idx + 1) % len(polygon)])
        for idx in range(len(polygon))
    ]
    return min(distances) if distances else float('inf')


def classify_with_polygon(point, config):
    polygon = [as_point(item) for item in config['court_polygon']]
    tolerance = float(config.get('line_tolerance_px', 3.0))
    inside = point_in_polygon(point, polygon)
    distance = distance_to_polygon(point, polygon)
    if distance <= tolerance:
        decision = 'IN'
        reason = 'on_line'
    elif inside:
        decision = 'IN'
        reason = 'inside_polygon'
    else:
        decision = 'OUT'
        reason = 'outside_polygon'
    return decision, reason, distance, ''


def classify_with_line(point, config):
    line_start = as_point(config['line_start'])
    line_end = as_point(config['line_end'])
    inside_point = as_point(config['inside_point'])
    tolerance = float(config.get('line_tolerance_px', 3.0))

    point_side = signed_distance_to_line(point, line_start, line_end)
    inside_side = signed_distance_to_line(inside_point, line_start, line_end)
    distance = abs(point_side)
    if distance <= tolerance:
        decision = 'IN'
        reason = 'on_line'
    elif point_side == 0 or inside_side == 0:
        decision = 'UNKNOWN'
        reason = 'invalid_side_reference'
    elif (point_side > 0) == (inside_side > 0):
        decision = 'IN'
        reason = 'same_side_as_inside_point'
    else:
        decision = 'OUT'
        reason = 'opposite_side_from_inside_point'
    return decision, reason, distance, point_side


def classify_point(point, config):
    mode = config.get('mode', 'polygon')
    if mode == 'polygon':
        return classify_with_polygon(point, config)
    if mode == 'line':
        return classify_with_line(point, config)
    raise ValueError(f'Unsupported config mode: {mode}')


def get_bounce_point(row, x_column, y_column):
    if x_column not in row or y_column not in row:
        raise KeyError(f'Missing coordinate columns: {x_column}, {y_column}')
    return float(row[x_column]), float(row[y_column])


def judge_csv(input_csv, output_csv, config, x_column, y_column):
    rows = read_csv(input_csv)
    output_rows = []
    for row in rows:
        point = get_bounce_point(row, x_column, y_column)
        decision, reason, boundary_distance, signed_distance = classify_point(point, config)
        output = dict(row)
        output['judge_x'] = round(point[0], 3)
        output['judge_y'] = round(point[1], 3)
        output['decision'] = decision
        output['decision_reason'] = reason
        output['boundary_distance_px'] = round(float(boundary_distance), 3)
        output['signed_distance_px'] = (
            '' if signed_distance == '' else round(float(signed_distance), 3)
        )
        output_rows.append(output)

    fieldnames = list(rows[0].keys()) if rows else []
    for name in [
        'judge_x', 'judge_y', 'decision', 'decision_reason',
        'boundary_distance_px', 'signed_distance_px',
    ]:
        if name not in fieldnames:
            fieldnames.append(name)
    write_csv(output_csv, output_rows, fieldnames)
    return len(output_rows)


def iter_bounce_csvs(input_path):
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.rglob('*_bounces.csv'))


def output_path_for(input_csv, input_root, output_root):
    if input_root.is_file():
        return output_root / f'{input_csv.stem}_judged.csv'
    rel_path = input_csv.relative_to(input_root)
    return output_root / rel_path.parent / f'{input_csv.stem}_judged.csv'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True,
                        help='bounce CSV file or folder containing *_bounces.csv')
    parser.add_argument('--court_config', type=Path, required=True,
                        help='JSON config describing court polygon or in/out line')
    parser.add_argument('--output_root', type=Path, default=Path('output_inout'))
    parser.add_argument('--x_column', type=str, default='contact_x')
    parser.add_argument('--y_column', type=str, default='contact_y')
    parser.add_argument('--config_image', type=str,
                        help='image name to select when court_config contains multiple view2 records')
    parser.add_argument('--config_index', type=int, default=0,
                        help='record index to select when court_config contains multiple view2 records')
    parser.add_argument('--target_width', type=float,
                        help='scale view2 line coordinates to this frame width')
    parser.add_argument('--target_height', type=float,
                        help='scale view2 line coordinates to this frame height')
    parser.add_argument('--video_path', type=Path,
                        help='video used to infer target width/height for view2 line configs')
    args = parser.parse_args()

    config = normalize_config(load_json(args.court_config), args)
    if config.get('source_mode') == 'view2_lines':
        print(
            'court_config=view2_lines image={} source_size={} target_size={}'.format(
                config.get('source_image', ''),
                config.get('source_size', ''),
                config.get('target_size', ''),
            )
        )
    csv_paths = iter_bounce_csvs(args.input)
    args.output_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    for csv_path in csv_paths:
        out_path = output_path_for(csv_path, args.input, args.output_root)
        count = judge_csv(csv_path, out_path, config, args.x_column, args.y_column)
        processed += 1
        print(f'{csv_path} -> {out_path} ({count} rows)')

    print(f'processed={processed}')


if __name__ == '__main__':
    main()
