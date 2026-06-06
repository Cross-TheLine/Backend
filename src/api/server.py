from __future__ import annotations

import csv
import json
import shutil
import uuid
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.bounce_detection.detect_bounces import (
    load_model,
    scale_track_to_frame,
    select_device,
    track_ball,
)
from src.bounce_detection.detect_bounces_from_track_csv import (
    detect_y_reversal_bounces,
    write_bounce_csv,
    write_track_csv,
    write_video as write_bounce_video,
)
from src.ball_tracking.infer_on_video import read_video
from src.line_detection.detect_view2_apriltag_lines import APRILTAG_FAMILIES, family_dictionary


OUTPUT_ROOT = Path("output") / "api_sessions"
DEFAULT_MODEL_PATH = Path("weights") / "tracknet_pretrained.pt"
DEFAULT_DEVICE = "auto"
DEFAULT_LOOKBACK_SEC = 3.0

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="Cross The Line Backend", version="0.1.0")
app.mount("/files", StaticFiles(directory=str(OUTPUT_ROOT), html=False), name="files")

_model = None
_model_device = None


class SessionStartRequest(BaseModel):
    camera_label: str | None = None


class SessionResponse(BaseModel):
    session_id: str
    status: str
    camera_label: str | None = None
    recording_path: str | None = None


class RecordPathRequest(BaseModel):
    path: str


class JudgeRequest(BaseModel):
    pressed_at_sec: float = Field(ge=0)
    lookback_sec: float = Field(default=DEFAULT_LOOKBACK_SEC, gt=0)
    render_video: bool = True


class JudgePreprocessRequest(BaseModel):
    recording_path: str
    pressed_at_sec: float = Field(ge=0)
    lookback_sec: float = Field(default=DEFAULT_LOOKBACK_SEC, gt=0)
    render_video: bool = True
    session_id: str | None = None


class JudgeQueuedResponse(BaseModel):
    job_id: str
    status: str


def session_dir(session_id: str) -> Path:
    return OUTPUT_ROOT / session_id


def session_meta_path(session_id: str) -> Path:
    return session_dir(session_id) / "session.json"


def job_dir(session_id: str, job_id: str) -> Path:
    return session_dir(session_id) / "jobs" / job_id


def job_meta_path(session_id: str, job_id: str) -> Path:
    return job_dir(session_id, job_id) / "job.json"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_session(session_id: str) -> dict[str, Any]:
    return read_json(session_meta_path(session_id))


def write_session(session: dict[str, Any]) -> None:
    write_json(session_meta_path(session["session_id"]), session)


def write_job(session_id: str, job: dict[str, Any]) -> None:
    write_json(job_meta_path(session_id, job["job_id"]), job)


def relative_file_url(path: Path) -> str:
    rel = path.resolve().relative_to(OUTPUT_ROOT.resolve())
    return "/files/" + rel.as_posix()


def get_model():
    global _model, _model_device
    if _model is None:
        if not DEFAULT_MODEL_PATH.exists():
            raise RuntimeError(f"missing model weights: {DEFAULT_MODEL_PATH}")
        _model_device = select_device(DEFAULT_DEVICE)
        _model = load_model(DEFAULT_MODEL_PATH, _model_device)
    return _model


def video_info(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    info = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def save_upload(upload: UploadFile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out_file:
        shutil.copyfileobj(upload.file, out_file)


def clip_video(source: Path, target: Path, start_sec: float, end_sec: float) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"could not open source video: {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, int(round(start_sec * fps)))
    end_frame = min(total_frames, int(round(end_sec * fps)))
    if end_frame <= start_frame:
        raise RuntimeError("clip range is empty")

    target.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    for _ in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    cap.release()
    writer.release()
    if written == 0:
        raise RuntimeError("no frames were written to clip")
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "start_frame": start_frame,
        "end_frame": start_frame + written,
        "frame_count": written,
    }


def tracking_args(render_video: bool) -> SimpleNamespace:
    return SimpleNamespace(
        min_confidence=82,
        max_prediction_gap=18,
        max_dist=190,
        max_gate_dist=480,
        min_gate_detections=4,
        offline_window=7,
        offline_polyorder=2,
        centroid_radius=9,
        relative_threshold=0.55,
        max_candidates=5,
        max_candidate_dist=340,
        score_distance_tradeoff=0.35,
        no_visual_refine=False,
        visual_roi_radius=36,
        no_fast_ball=False,
        fast_speed_threshold=50,
        fast_min_confidence=55,
        max_bridge_gap=6,
        suppress_isolated=True,
        isolation_window=2,
        isolation_max_dist=220,
        suppress_jumps=False,
        jump_window=6,
        jump_max_interp_error=90,
        jump_max_step_dist=190,
        window=3,
        local_window=4,
        min_local_points=5,
        min_down_speed=3.0,
        min_up_speed=2.0,
        local_y_tolerance=10.0,
        max_x_velocity_change=85.0,
        min_score=9.0,
        min_gap=10,
        trace=7,
        bounce_display_window=3,
        hide_track=False,
        render_video=render_video,
    )


def bounce_payload(row: dict[str, Any], clip_start_sec: float) -> dict[str, Any]:
    return {
        "frame_index": int(row["frame_index"]),
        "clip_time_sec": float(row["time_sec"]),
        "recording_time_sec": float(row["time_sec"]) + clip_start_sec,
        "x": float(row["x"]),
        "y": float(row["y"]),
        "score": float(row.get("score", 0.0)),
        "vy_before": float(row.get("vy_before", 0.0)),
        "vy_after": float(row.get("vy_after", 0.0)),
        "prominence": float(row.get("prominence", 0.0)),
        "x_velocity_change": float(row.get("x_velocity_change", 0.0)),
    }


def run_preprocess(
    source_path: Path,
    current_job_dir: Path,
    pressed_at_sec: float,
    lookback_sec: float,
    render_video: bool,
) -> dict[str, Any]:
    start_sec = max(0.0, pressed_at_sec - lookback_sec)
    end_sec = pressed_at_sec

    clip_path = current_job_dir / "clip.mp4"
    clip_info = clip_video(source_path, clip_path, start_sec, end_sec)

    frames, fps = read_video(str(clip_path))
    if not frames:
        raise RuntimeError("clip has no readable frames")

    args = tracking_args(render_video=render_video)
    model = get_model()
    track, statuses, scores, raw_detected, after_outlier = track_ball(frames, model, args)
    frame_height, frame_width = frames[0].shape[:2]
    video_track = scale_track_to_frame(track, frame_width, frame_height)
    bounces = detect_y_reversal_bounces(video_track, fps, args)

    track_csv = current_job_dir / "track.csv"
    bounce_csv = current_job_dir / "bounces.csv"
    result_video = current_job_dir / "result.avi"
    write_track_csv(track_csv, video_track, statuses, scores, fps)
    write_bounce_csv(bounce_csv, bounces)
    if render_video:
        write_bounce_video(frames, video_track, bounces, result_video, fps)

    counts = Counter(statuses)
    selected = max(bounces, key=lambda row: float(row.get("score", 0.0)), default=None)
    bounces_payload = [bounce_payload(row, start_sec) for row in bounces]
    primary_bounce = bounce_payload(selected, start_sec) if selected is not None else None
    confidence = 0.0 if selected is None else min(1.0, max(0.0, float(selected.get("score", 0.0)) / 150.0))

    artifacts = {
        "clip": relative_file_url(clip_path),
        "track_csv": relative_file_url(track_csv),
        "bounces_csv": relative_file_url(bounce_csv),
    }
    if result_video.exists():
        artifacts["result_video"] = relative_file_url(result_video)

    return {
        "clip": {
            "start_sec": start_sec,
            "end_sec": end_sec,
            **clip_info,
        },
        "result": "bounce_detected" if primary_bounce is not None else "unknown",
        "confidence": confidence,
        "primary_bounce": primary_bounce,
        "bounces": bounces_payload,
        "inout_input": {
            "fps": float(fps),
            "frame_width": frame_width,
            "frame_height": frame_height,
            "clip_start_sec": start_sec,
            "bounces": bounces_payload,
            "track_csv": str(track_csv),
            "bounces_csv": str(bounce_csv),
        },
        "tracking": {
            "frames": len(frames),
            "raw_detected": raw_detected,
            "after_outlier": after_outlier,
            "detected": counts.get("detected", 0),
            "predicted": counts.get("predicted", 0),
            "isolated": counts.get("isolated", 0),
            "missing": counts.get("missing", 0),
            "bounce_count": len(bounces),
        },
        "artifacts": artifacts,
    }


def run_judgement_job(session_id: str, job_id: str) -> None:
    job = read_json(job_meta_path(session_id, job_id))
    try:
        job["status"] = "running"
        write_job(session_id, job)

        session = read_session(session_id)
        source_path = Path(session["recording_path"])
        result = run_preprocess(
            source_path=source_path,
            current_job_dir=job_dir(session_id, job_id),
            pressed_at_sec=float(job["pressed_at_sec"]),
            lookback_sec=float(job["lookback_sec"]),
            render_video=bool(job.get("render_video", True)),
        )
        job.update({"status": "done", **result})
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
    write_job(session_id, job)


@app.post("/sessions/start", response_model=SessionResponse)
def start_session(request: SessionStartRequest) -> SessionResponse:
    session_id = str(uuid.uuid4())
    session = {
        "session_id": session_id,
        "status": "created",
        "camera_label": request.camera_label,
        "recording_path": None,
    }
    write_session(session)
    return SessionResponse(**session)


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    return read_session(session_id)


@app.post("/sessions/{session_id}/line-status")
def line_status(
    session_id: str,
    frame: UploadFile = File(...),
    family: str = "tag36h11",
) -> dict[str, Any]:
    if family not in APRILTAG_FAMILIES:
        raise HTTPException(status_code=400, detail=f"unknown family: {family}")
    session = read_session(session_id)
    frame_path = session_dir(session_id) / "line_checks" / f"{uuid.uuid4()}.jpg"
    save_upload(frame, frame_path)

    image = cv2.imread(str(frame_path))
    if image is None:
        raise HTTPException(status_code=400, detail="could not read uploaded frame")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(family_dictionary(family))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    _, ids, rejected = detector.detectMarkers(gray)
    marker_count = 0 if ids is None else len(ids)
    rejected_count = 0 if rejected is None else len(rejected)
    ready = marker_count >= 3
    response = {
        "ready": ready,
        "line_visible": ready,
        "marker_visible": marker_count > 0,
        "marker_count": marker_count,
        "rejected_count": rejected_count,
        "confidence": min(1.0, marker_count / 3.0),
        "frame": relative_file_url(frame_path),
    }
    session["last_line_status"] = response
    write_session(session)
    return response


@app.post("/sessions/{session_id}/record/start")
def record_start(session_id: str) -> dict[str, str]:
    session = read_session(session_id)
    session["status"] = "recording"
    write_session(session)
    return {"session_id": session_id, "status": "recording"}


@app.post("/sessions/{session_id}/record/upload")
def upload_recording(session_id: str, video: UploadFile = File(...)) -> dict[str, Any]:
    session = read_session(session_id)
    suffix = Path(video.filename or "recording.mp4").suffix or ".mp4"
    recording_path = session_dir(session_id) / f"recording{suffix}"
    save_upload(video, recording_path)
    info = video_info(recording_path)
    session["recording_path"] = str(recording_path)
    session["recording"] = {
        **info,
        "url": relative_file_url(recording_path),
    }
    session["status"] = "recorded"
    write_session(session)
    return {"session_id": session_id, "recording": session["recording"]}


@app.post("/sessions/{session_id}/record/path")
def set_recording_path(session_id: str, request: RecordPathRequest) -> dict[str, Any]:
    session = read_session(session_id)
    recording_path = Path(request.path)
    if not recording_path.exists():
        raise HTTPException(status_code=404, detail=f"recording not found: {recording_path}")
    session["recording_path"] = str(recording_path)
    session["recording"] = video_info(recording_path)
    session["status"] = "recorded"
    write_session(session)
    return {"session_id": session_id, "recording": session["recording"]}


@app.post("/sessions/{session_id}/record/stop")
def record_stop(session_id: str) -> dict[str, str]:
    session = read_session(session_id)
    session["status"] = "stopped"
    write_session(session)
    return {"session_id": session_id, "status": "stopped"}


@app.post("/sessions/{session_id}/judge", response_model=JudgeQueuedResponse)
def judge(session_id: str, request: JudgeRequest, background_tasks: BackgroundTasks) -> JudgeQueuedResponse:
    session = read_session(session_id)
    if not session.get("recording_path"):
        raise HTTPException(status_code=409, detail="session has no recording")
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "session_id": session_id,
        "status": "pending",
        "pressed_at_sec": request.pressed_at_sec,
        "lookback_sec": request.lookback_sec,
        "render_video": request.render_video,
    }
    write_job(session_id, job)
    background_tasks.add_task(run_judgement_job, session_id, job_id)
    return JudgeQueuedResponse(job_id=job_id, status="pending")


@app.post("/judge-preprocess", response_model=JudgeQueuedResponse)
def judge_preprocess(request: JudgePreprocessRequest, background_tasks: BackgroundTasks) -> JudgeQueuedResponse:
    recording_path = Path(request.recording_path)
    if not recording_path.exists():
        raise HTTPException(status_code=404, detail=f"recording not found: {recording_path}")

    session_id = request.session_id or str(uuid.uuid4())
    session = {
        "session_id": session_id,
        "status": "recorded",
        "camera_label": None,
        "recording_path": str(recording_path),
        "recording": video_info(recording_path),
    }
    write_session(session)

    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "session_id": session_id,
        "status": "pending",
        "pressed_at_sec": request.pressed_at_sec,
        "lookback_sec": request.lookback_sec,
        "render_video": request.render_video,
    }
    write_job(session_id, job)
    background_tasks.add_task(run_judgement_job, session_id, job_id)
    return JudgeQueuedResponse(job_id=job_id, status="pending")


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    for path in OUTPUT_ROOT.glob(f"*/jobs/{job_id}/job.json"):
        return read_json(path)
    raise HTTPException(status_code=404, detail=f"unknown job_id: {job_id}")


@app.post("/sessions/{session_id}/save")
def save_session(session_id: str) -> dict[str, str]:
    session = read_session(session_id)
    session["saved"] = True
    write_session(session)
    return {"session_id": session_id, "status": "saved"}


@app.post("/sessions/{session_id}/finish")
def finish_session(session_id: str) -> dict[str, str]:
    session = read_session(session_id)
    session["status"] = "finished"
    write_session(session)
    return {"session_id": session_id, "status": "finished"}
