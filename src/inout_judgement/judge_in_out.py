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
    args = parser.parse_args()

    config = load_json(args.court_config)
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
