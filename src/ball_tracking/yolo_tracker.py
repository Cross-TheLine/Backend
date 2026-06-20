from __future__ import annotations

from typing import Any


YOLO_TRACK_FIELDNAMES = [
    "frame_index",
    "time_sec",
    "status",
    "class_id",
    "confidence",
    "x",
    "y",
    "x1",
    "y1",
    "x2",
    "y2",
]


def choose_yolo_detection(result: Any, class_id: int | None) -> dict[str, Any] | None:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    best = None
    best_confidence = -1.0
    for box in boxes:
        detected_class_id = int(box.cls[0].item()) if box.cls is not None else -1
        if class_id is not None and detected_class_id != class_id:
            continue
        confidence = float(box.conf[0].item()) if box.conf is not None else 0.0
        if confidence <= best_confidence:
            continue
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        best = {
            "class_id": detected_class_id,
            "confidence": confidence,
            "x": (x1 + x2) * 0.5,
            "y": (y1 + y2) * 0.5,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }
        best_confidence = confidence
    return best


def detect_ball_yolo(
    frame: Any,
    model: Any,
    *,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    max_det: int,
    class_id: int | None,
) -> dict[str, float] | None:
    result = model.predict(
        source=frame,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        max_det=max_det,
        verbose=False,
    )[0]
    return choose_yolo_detection(result, class_id)


def detection_to_track_row(
    frame_index: int,
    fps: float,
    detection: dict[str, Any] | None,
) -> dict[str, Any]:
    if detection is None:
        return {
            "frame_index": frame_index,
            "time_sec": frame_index / fps if fps else 0.0,
            "status": "missing",
            "class_id": "",
            "confidence": "",
            "x": "",
            "y": "",
            "x1": "",
            "y1": "",
            "x2": "",
            "y2": "",
        }

    return {
        "frame_index": frame_index,
        "time_sec": frame_index / fps if fps else 0.0,
        "status": "detected",
        **detection,
    }


def track_frames_yolo(
    frames: list[Any],
    model: Any,
    *,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    max_det: int,
    class_id: int | None,
) -> tuple[list[tuple[float | None, float | None]], list[str], list[float], int]:
    track: list[tuple[float | None, float | None]] = []
    statuses: list[str] = []
    scores: list[float] = []
    for frame in frames:
        detection = detect_ball_yolo(
            frame,
            model,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            max_det=max_det,
            class_id=class_id,
        )
        if detection is None:
            track.append((None, None))
            statuses.append("missing")
            scores.append(0.0)
        else:
            track.append((float(detection["x"]), float(detection["y"])))
            statuses.append("detected")
            scores.append(float(detection["confidence"]))
    return track, statuses, scores, statuses.count("detected")
