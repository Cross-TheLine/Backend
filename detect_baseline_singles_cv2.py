from __future__ import annotations

import argparse
import base64
import json
import math
import os
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class LineCluster:
    angle_deg: float
    rho: float
    weight: float
    support: int
    p1: tuple[float, float]
    p2: tuple[float, float]
    length: float


def roboflow_detect(
    image_path: Path,
    model_id: str | None,
    api_key: str,
    api_url: str = "https://serverless.roboflow.com",
    client_mode: str = "sdk",
    workspace_name: str | None = None,
    workflow_id: str | None = None,
    workflow_image_input: str = "image",
    confidence: int = 30,
    overlap: int = 30,
) -> dict:
    if client_mode == "sdk":
        if not workflow_id and not model_id:
            raise RuntimeError("Roboflow SDK mode requires either model_id or workflow_id.")
        try:
            from inference_sdk import InferenceHTTPClient
        except ImportError as exc:
            raise RuntimeError(
                "inference-sdk가 설치되어 있지 않습니다. "
                "`conda run -n courtcv python -m pip install inference-sdk --no-deps` 후 필요한 의존성을 설치하세요."
            ) from exc

        client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
        try:
            if workflow_id:
                result = client.run_workflow(
                    workspace_name=workspace_name,
                    workflow_id=workflow_id,
                    images={workflow_image_input: str(image_path)},
                    use_cache=True,
                )
            else:
                result = client.infer(str(image_path), model_id=model_id)
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            api_message = getattr(exc, "api_message", None)
            detail = f"status={status_code}, message={api_message}" if status_code else str(exc)
            raise RuntimeError(f"Roboflow SDK 호출 실패: {redact_secret(detail, api_key)}") from exc
        if isinstance(result, list):
            return {"workflow_outputs": result, "predictions": extract_predictions(result)}
        return result

    if not model_id:
        raise RuntimeError("Roboflow legacy mode requires model_id.")
    encoded = base64.b64encode(image_path.read_bytes())
    query = urllib.parse.urlencode(
        {
            "api_key": api_key,
            "confidence": confidence,
            "overlap": overlap,
            "name": image_path.name,
        }
    )
    url = f"https://detect.roboflow.com/{model_id}?{query}"
    request = urllib.request.Request(
        url,
        data=encoded,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        safe_body = redact_secret(body, api_key)
        raise RuntimeError(f"Roboflow legacy HTTP 호출 실패: HTTP {exc.code} {exc.reason}: {safe_body}") from exc


def redact_secret(text: object, secret: str | None) -> str:
    value = str(text)
    if secret:
        value = value.replace(secret, "<redacted>")
    return value


def bbox_from_prediction(prediction: dict, image_shape: tuple[int, int, int], margin: int) -> tuple[int, int, int, int]:
    h, w = image_shape[:2]
    x = float(prediction["x"])
    y = float(prediction["y"])
    width = float(prediction["width"])
    height = float(prediction["height"])
    x1 = max(0, int(round(x - width / 2 - margin)))
    y1 = max(0, int(round(y - height / 2 - margin)))
    x2 = min(w, int(round(x + width / 2 + margin)))
    y2 = min(h, int(round(y + height / 2 + margin)))
    return x1, y1, x2, y2


def extract_predictions(value) -> list[dict]:
    found: list[dict] = []
    if isinstance(value, list):
        for item in value:
            found.extend(extract_predictions(item))
    elif isinstance(value, dict):
        predictions = value.get("predictions")
        if isinstance(predictions, list):
            for prediction in predictions:
                if isinstance(prediction, dict):
                    if {"x", "y", "width", "height"}.issubset(prediction.keys()):
                        found.append(prediction)
                    else:
                        found.extend(extract_predictions(prediction))
        for key, item in value.items():
            if key != "predictions":
                found.extend(extract_predictions(item))
    return found


def select_court_prediction(result: dict) -> dict | None:
    court_predictions = [
        prediction
        for prediction in extract_predictions(result)
        if str(prediction.get("class", "")).lower() == "court"
    ]
    if not court_predictions:
        return None
    return max(
        court_predictions,
        key=lambda prediction: float(prediction.get("confidence", 0.0))
        * float(prediction.get("width", 0.0))
        * float(prediction.get("height", 0.0)),
    )


def apply_bbox_roi(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    roi_mask = np.zeros_like(mask)
    roi_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return roi_mask


def draw_roi_box(image: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    out = image.copy()
    if bbox is None:
        return out
    x1, y1, x2, y2 = bbox
    overlay = out.copy()
    overlay[:y1, :] = 0
    overlay[y2:, :] = 0
    overlay[:, :x1] = 0
    overlay[:, x2:] = 0
    out = cv2.addWeighted(overlay, 0.82, out, 0.18, 0)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(out, "roboflow court ROI", (x1 + 10, max(30, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return out


def normalize_angle(angle_deg: float) -> float:
    while angle_deg > 90.0:
        angle_deg -= 180.0
    while angle_deg <= -90.0:
        angle_deg += 180.0
    return angle_deg


def segment_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    return normalize_angle(math.degrees(math.atan2(y2 - y1, x2 - x1)))


def line_normal_from_angle(angle_deg: float) -> tuple[float, float]:
    theta = math.radians(angle_deg + 90.0)
    return math.cos(theta), math.sin(theta)


def canonical_line(
    x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float, float, float]:
    angle = segment_angle(x1, y1, x2, y2)
    nx, ny = line_normal_from_angle(angle)
    rho = nx * x1 + ny * y1
    if rho < 0:
        rho = -rho
        angle = normalize_angle(angle + 180.0)
        nx, ny = -nx, -ny
    length = math.hypot(x2 - x1, y2 - y1)
    return angle, rho, nx, ny, length


def make_white_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    b, g, r = cv2.split(image)
    spread = np.maximum.reduce([b, g, r]) - np.minimum.reduce([b, g, r])

    mask = (
        (v > 155)
        & (s < 82)
        & (spread < 95)
        & (b > 115)
        & (g > 115)
        & (r > 115)
    ).astype(np.uint8) * 255

    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
    return mask


def hough_segments(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=58,
        minLineLength=70,
        maxLineGap=24,
    )
    if raw is None:
        return []
    return [tuple(map(int, line[0])) for line in raw]


def cluster_segments(
    segments: list[tuple[int, int, int, int]], mask: np.ndarray
) -> list[LineCluster]:
    buckets: list[list[tuple[float, float, float, tuple[int, int, int, int]]]] = []
    for x1, y1, x2, y2 in segments:
        angle, rho, _, _, length = canonical_line(x1, y1, x2, y2)
        if length < 70:
            continue
        placed = False
        for bucket in buckets:
            b_angle = np.average([item[0] for item in bucket], weights=[item[2] for item in bucket])
            b_rho = np.average([item[1] for item in bucket], weights=[item[2] for item in bucket])
            if abs(angle - b_angle) < 5.0 and abs(rho - b_rho) < 24.0:
                bucket.append((angle, rho, length, (x1, y1, x2, y2)))
                placed = True
                break
        if not placed:
            buckets.append([(angle, rho, length, (x1, y1, x2, y2))])

    clusters: list[LineCluster] = []
    ys, xs = np.nonzero(mask)
    for bucket in buckets:
        weights = np.array([item[2] for item in bucket], dtype=np.float64)
        angle = float(np.average([item[0] for item in bucket], weights=weights))
        rho = float(np.average([item[1] for item in bucket], weights=weights))
        nx, ny = line_normal_from_angle(angle)
        distances = np.abs(xs * nx + ys * ny - rho)
        support_mask = distances < 8.0
        support = int(support_mask.sum())
        if support < 95:
            continue

        direction = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
        sx = xs[support_mask]
        sy = ys[support_mask]
        t = sx * direction[0] + sy * direction[1]
        lo = float(np.percentile(t, 2))
        hi = float(np.percentile(t, 98))
        length = hi - lo
        if length < 95:
            continue
        base = np.array([nx * rho, ny * rho])
        p1 = base + direction * lo
        p2 = base + direction * hi
        clusters.append(
            LineCluster(
                angle_deg=angle,
                rho=rho,
                weight=float(weights.sum()),
                support=support,
                p1=(float(p1[0]), float(p1[1])),
                p2=(float(p2[0]), float(p2[1])),
                length=float(length),
            )
        )

    clusters.sort(key=lambda c: c.length * math.sqrt(c.support), reverse=True)
    return merge_duplicate_clusters(clusters)


def merge_duplicate_clusters(clusters: list[LineCluster]) -> list[LineCluster]:
    kept: list[LineCluster] = []
    for cluster in clusters:
        duplicate = False
        for old in kept:
            if abs(cluster.angle_deg - old.angle_deg) < 7.5 and abs(cluster.rho - old.rho) < 42.0:
                duplicate = True
                break
        if not duplicate:
            kept.append(cluster)
    return kept


def x_at_y(line: LineCluster, y: float) -> float | None:
    x1, y1 = line.p1
    x2, y2 = line.p2
    dy = y2 - y1
    if abs(dy) < 1e-6:
        return None
    t = (y - y1) / dy
    return x1 + t * (x2 - x1)


def bottom_x(line: LineCluster, image_h: int) -> float:
    x = x_at_y(line, image_h - 1)
    if x is not None:
        return x
    return (line.p1[0] + line.p2[0]) / 2.0


def point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def endpoint_distance(line: LineCluster, point: tuple[float, float]) -> float:
    return min(point_distance(line.p1, point), point_distance(line.p2, point))


def baseline_endpoints(line: LineCluster) -> tuple[tuple[float, float], tuple[float, float]]:
    return (line.p1, line.p2) if line.p1[0] <= line.p2[0] else (line.p2, line.p1)


def classify_lines(clusters: list[LineCluster], image_shape: tuple[int, int, int]) -> dict[str, LineCluster]:
    h, w = image_shape[:2]

    baseline_candidates = [
        line
        for line in clusters
        if 5.0 <= line.angle_deg <= 35.0
        and max(line.p1[1], line.p2[1]) > h * 0.72
        and line.length > w * 0.28
    ]
    if not baseline_candidates:
        raise RuntimeError("baseline 후보를 찾지 못했습니다.")

    baseline = max(
        baseline_candidates,
        key=lambda line: line.length * math.sqrt(line.support) + max(line.p1[1], line.p2[1]) * 4.0,
    )

    singles_candidates = [
        line
        for line in clusters
        if -78.0 <= line.angle_deg <= -25.0
        and line.length > h * 0.45
        and max(line.p1[1], line.p2[1]) > h * 0.76
    ]
    singles_candidates.sort(key=lambda line: line.length * math.sqrt(line.support), reverse=True)

    singles: list[LineCluster] = []
    for candidate in singles_candidates:
        bx = bottom_x(candidate, h)
        separated = all(abs(bx - bottom_x(old, h)) > w * 0.18 for old in singles)
        if separated:
            singles.append(candidate)
        if len(singles) == 2:
            break

    if len(singles) < 2:
        left_endpoint, right_endpoint = baseline_endpoints(baseline)
        join_radius = max(85.0, w * 0.08)
        side_candidates = [
            line
            for line in clusters
            if line is not baseline
            and -80.0 <= line.angle_deg <= 12.0
            and line.length > min(w, h) * 0.48
            and max(line.p1[1], line.p2[1]) > h * 0.56
        ]

        left_joined = [
            line
            for line in side_candidates
            if endpoint_distance(line, left_endpoint) <= join_radius
        ]
        right_joined = [
            line
            for line in side_candidates
            if endpoint_distance(line, right_endpoint) <= join_radius
        ]

        if left_joined and right_joined:
            left_line = max(
                left_joined,
                key=lambda line: line.length * math.sqrt(line.support)
                - endpoint_distance(line, left_endpoint) * 15.0,
            )
            right_line = max(
                right_joined,
                key=lambda line: line.length * math.sqrt(line.support)
                - endpoint_distance(line, right_endpoint) * 15.0,
            )
            singles = [left_line, right_line]

    if len(singles) < 2:
        raise RuntimeError("싱글라인 후보 2개를 찾지 못했습니다.")

    singles.sort(key=lambda line: bottom_x(line, h))
    return {
        "baseline": baseline,
        "single_line_left": singles[0],
        "single_line_right": singles[1],
    }


def draw_all_clusters(image: np.ndarray, clusters: list[LineCluster]) -> np.ndarray:
    out = image.copy()
    colors = [
        (40, 40, 255),
        (255, 210, 0),
        (0, 220, 255),
        (80, 255, 80),
        (255, 80, 255),
        (0, 130, 255),
    ]
    for index, line in enumerate(clusters[:28]):
        color = colors[index % len(colors)]
        p1 = tuple(np.round(line.p1).astype(int))
        p2 = tuple(np.round(line.p2).astype(int))
        cv2.line(out, p1, p2, color, 3, cv2.LINE_AA)
        mx = int((p1[0] + p2[0]) / 2)
        my = int((p1[1] + p2[1]) / 2)
        cv2.putText(out, str(index), (mx + 4, my + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return out


def draw_selected(image: np.ndarray, selected: dict[str, LineCluster]) -> np.ndarray:
    out = image.copy()
    styles = {
        "baseline": ((0, 220, 255), 8),
        "single_line_left": ((40, 40, 255), 7),
        "single_line_right": ((40, 255, 40), 7),
    }
    for name, line in selected.items():
        color, width = styles[name]
        p1 = tuple(np.round(line.p1).astype(int))
        p2 = tuple(np.round(line.p2).astype(int))
        cv2.line(out, p1, p2, color, width, cv2.LINE_AA)
        mx = int((p1[0] + p2[0]) / 2)
        my = int((p1[1] + p2[1]) / 2)
        cv2.putText(out, name, (mx + 8, my - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def serialize_line(line: LineCluster) -> dict:
    return {
        "angle_deg": round(line.angle_deg, 3),
        "rho": round(line.rho, 3),
        "support": line.support,
        "length": round(line.length, 3),
        "p1": [round(line.p1[0], 2), round(line.p1[1], 2)],
        "p2": [round(line.p2[0], 2), round(line.p2[1], 2)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect tennis court baseline and two singles sidelines from an image."
    )
    parser.add_argument("--input", required=True, help="Path to the input image.")
    parser.add_argument("--out-dir", default="line_detection_output", help="Directory for result images and JSON.")
    parser.add_argument("--use-roboflow-roi", action="store_true")
    parser.add_argument("--roboflow-model", default="tennis-vhrs9/9", help="Roboflow model id.")
    parser.add_argument("--roboflow-api-key", default=None, help="Roboflow API key. Prefer ROBOFLOW_API_KEY env var.")
    parser.add_argument("--roboflow-api-url", default="https://serverless.roboflow.com")
    parser.add_argument("--roboflow-client", choices=["sdk", "legacy"], default="sdk")
    parser.add_argument("--roboflow-workspace", default=None)
    parser.add_argument("--roboflow-workflow-id", default=None)
    parser.add_argument("--roboflow-workflow-image-input", default="image")
    parser.add_argument("--roboflow-margin", type=int, default=30)
    parser.add_argument("--roboflow-confidence", type=int, default=25)
    args = parser.parse_args()

    if args.use_roboflow_roi and not args.roboflow_model and not args.roboflow_workflow_id:
        parser.error("--use-roboflow-roi requires --roboflow-model or --roboflow-workflow-id.")

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(input_path)

    mask = make_white_mask(image)
    roboflow_result = None
    court_bbox = None
    roboflow_warning = None
    if args.use_roboflow_roi:
        api_key = args.roboflow_api_key or os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Roboflow ROI를 쓰려면 --roboflow-api-key 또는 ROBOFLOW_API_KEY 환경변수가 필요합니다."
            )
        roboflow_result = roboflow_detect(
            input_path,
            model_id=args.roboflow_model,
            api_key=api_key,
            api_url=args.roboflow_api_url,
            client_mode=args.roboflow_client,
            workspace_name=args.roboflow_workspace,
            workflow_id=args.roboflow_workflow_id,
            workflow_image_input=args.roboflow_workflow_image_input,
            confidence=args.roboflow_confidence,
        )
        court_prediction = select_court_prediction(roboflow_result)
        if court_prediction is None:
            roboflow_warning = "Roboflow 결과에서 class='court' bbox를 찾지 못해 ROI 없이 전체 이미지로 진행했습니다."
        else:
            court_bbox = bbox_from_prediction(court_prediction, image.shape, args.roboflow_margin)
            mask = apply_bbox_roi(mask, court_bbox)

    segments = hough_segments(mask)
    clusters = cluster_segments(segments, mask)
    cv2.imwrite(str(out_dir / "01_white_mask.png"), mask)
    cv2.imwrite(str(out_dir / "00_roboflow_roi.png"), draw_roi_box(image, court_bbox))
    cv2.imwrite(str(out_dir / "02_all_line_clusters.png"), draw_all_clusters(image, clusters))

    selected: dict[str, LineCluster] = {}
    error: str | None = None
    try:
        selected = classify_lines(clusters, image.shape)
        cv2.imwrite(str(out_dir / "03_baseline_and_single_lines.png"), draw_selected(image, selected))
    except RuntimeError as exc:
        error = str(exc)

    data = {
        "input": input_path.name,
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "raw_hough_segments": len(segments),
        "line_clusters": len(clusters),
        "roboflow_model": args.roboflow_model if args.use_roboflow_roi else None,
        "roboflow_api_url": args.roboflow_api_url if args.use_roboflow_roi else None,
        "roboflow_client": args.roboflow_client if args.use_roboflow_roi else None,
        "roboflow_workspace": args.roboflow_workspace if args.use_roboflow_roi else None,
        "roboflow_workflow_id": args.roboflow_workflow_id if args.use_roboflow_roi else None,
        "roboflow_court_bbox": list(court_bbox) if court_bbox is not None else None,
        "roboflow_roi_applied": court_bbox is not None,
        "roboflow_warning": roboflow_warning,
        "roboflow_predictions": roboflow_result.get("predictions", []) if roboflow_result else None,
        "error": error,
        "selected": {name: serialize_line(line) for name, line in selected.items()},
        "clusters": [serialize_line(line) for line in clusters],
    }
    (out_dir / "detected_lines.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
