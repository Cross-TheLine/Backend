from __future__ import annotations

import base64
import json
import urllib.parse
import urllib.request
from pathlib import Path
from urllib.error import HTTPError

import numpy as np


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
