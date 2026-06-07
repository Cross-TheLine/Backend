from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import uuid
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import cv2
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.api.database import (
    encode_json,
    get_judgement_record,
    init_db,
    insert_judgement_record,
    list_judgement_records,
)
from src.inout_judgement.judge_in_out import judge_csv, load_json, normalize_config
from src.inout_judgement.overlay_in_out import write_overlay_video
from src.line_detection.detect_view2_apriltag_lines import (
    APRILTAG_FAMILIES,
    family_dictionary,
    process_image as detect_view2_court_config,
    read_image_exif,
)


OUTPUT_ROOT = Path("output") / "api_sessions"
DEFAULT_MODEL_PATH = Path("weights") / "tracknet_pretrained.pt"
DEFAULT_DEVICE = "auto"
DEFAULT_LOOKBACK_SEC = 2.0

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
init_db()
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


class CourtConfigPathRequest(BaseModel):
    path: str
    config_image: str | None = None
    config_index: int = 0


class JudgeRequest(BaseModel):
    pressed_at_sec: float | None = Field(default=None, ge=0)
    use_video_end: bool = False
    end_offset_sec: float = Field(default=0.0, ge=0)
    lookback_sec: float = Field(default=DEFAULT_LOOKBACK_SEC, gt=0)
    render_video: bool = True
    court_config_path: str | None = None
    config_image: str | None = None
    config_index: int | None = None
    render_inout_video: bool = True


class JudgePreprocessRequest(BaseModel):
    recording_path: str
    pressed_at_sec: float | None = Field(default=None, ge=0)
    use_video_end: bool = False
    end_offset_sec: float = Field(default=0.0, ge=0)
    lookback_sec: float = Field(default=DEFAULT_LOOKBACK_SEC, gt=0)
    render_video: bool = True
    session_id: str | None = None
    court_config_path: str | None = None
    config_image: str | None = None
    config_index: int = 0
    render_inout_video: bool = True


class JudgeQueuedResponse(BaseModel):
    job_id: str
    status: str


class ArtifactUrls(BaseModel):
    clip: str | None = None
    track_csv: str | None = None
    bounces_csv: str | None = None
    result_video: str | None = None
    inout_csv: str | None = None
    inout_overlay_video: str | None = None


class BounceResult(BaseModel):
    frame_index: int
    clip_time_sec: float
    recording_time_sec: float
    x: float
    y: float
    score: float | None = None
    vy_before: float | None = None
    vy_after: float | None = None
    prominence: float | None = None
    x_velocity_change: float | None = None


class InoutDecision(BaseModel):
    frame_index: int
    clip_time_sec: float
    recording_time_sec: float
    x: float
    y: float
    decision: str
    decision_reason: str
    boundary_distance_px: float | None = None
    signed_distance_px: float | None = None


class FrontendJobResult(BaseModel):
    job_id: str
    session_id: str | None = None
    status: str
    result: str | None = None
    confidence: float | None = None
    timing: dict[str, Any] | None = None
    primary_bounce: BounceResult | None = None
    primary_decision: InoutDecision | None = None
    bounces: list[BounceResult] = Field(default_factory=list)
    decisions: list[InoutDecision] = Field(default_factory=list)
    artifacts: ArtifactUrls = Field(default_factory=ArtifactUrls)
    error: str | None = None


class SaveJudgementRequest(BaseModel):
    match_type: Literal["singles", "doubles"]
    recorded_at: datetime | None = None
    recorded_date: date | None = None


class SavedJudgementResponse(BaseModel):
    id: str
    session_id: str | None = None
    job_id: str
    created_at: str
    recorded_at: str
    recorded_date: str | None = None
    match_type: Literal["singles", "doubles"]
    decision: str
    decision_reason: str | None = None
    video_path: str | None = None
    video_url: str | None = None
    clip_path: str | None = None
    clip_url: str | None = None
    result_video_url: str | None = None
    inout_overlay_video_url: str | None = None
    inout_csv_url: str | None = None
    confidence: float | None = None
    primary_bounce: dict[str, Any] | None = None
    primary_decision: dict[str, Any] | None = None


class SavedJudgementDetail(SavedJudgementResponse):
    job_result: dict[str, Any]


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


def find_job(job_id: str) -> dict[str, Any]:
    for path in OUTPUT_ROOT.glob(f"*/jobs/{job_id}/job.json"):
        return read_json(path)
    raise HTTPException(status_code=404, detail=f"unknown job_id: {job_id}")


def relative_file_url(path: Path) -> str:
    rel = path.resolve().relative_to(OUTPUT_ROOT.resolve())
    return "/files/" + rel.as_posix()


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def runtime_health() -> dict[str, Any]:
    weights_available = DEFAULT_MODEL_PATH.exists()
    return {
        "status": "ok" if torch_available() and weights_available else "degraded",
        "torch_available": torch_available(),
        "weights_available": weights_available,
        "weights_path": str(DEFAULT_MODEL_PATH),
        "weights_size_bytes": DEFAULT_MODEL_PATH.stat().st_size if weights_available else 0,
        "judge_ready": torch_available() and weights_available,
    }


def get_model():
    global _model, _model_device
    if _model is None:
        try:
            from src.bounce_detection.detect_bounces import load_model, select_device
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise RuntimeError(
                    "missing torch. Install PyTorch before running full judge jobs."
                ) from exc
            raise

        if not DEFAULT_MODEL_PATH.exists():
            raise RuntimeError(f"missing model weights: {DEFAULT_MODEL_PATH}")
        _model_device = select_device(DEFAULT_DEVICE)
        _model = load_model(DEFAULT_MODEL_PATH, _model_device)
    return _model


def video_info(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    duration_sec = frame_count / fps if fps > 0 else 0.0
    info = {
        "frame_count": frame_count,
        "fps": fps,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_sec": duration_sec,
    }
    cap.release()
    return info


def save_upload(upload: UploadFile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    upload.file.seek(0)
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


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        return list(csv.DictReader(csv_file))


def inout_payload(row: dict[str, Any], clip_start_sec: float) -> dict[str, Any]:
    return {
        "frame_index": int(float(row["frame_index"])),
        "clip_time_sec": float(row["time_sec"]),
        "recording_time_sec": float(row["time_sec"]) + clip_start_sec,
        "x": float(row.get("judge_x") or row.get("x")),
        "y": float(row.get("judge_y") or row.get("y")),
        "decision": row.get("decision", "UNKNOWN"),
        "decision_reason": row.get("decision_reason", ""),
        "boundary_distance_px": (
            None if row.get("boundary_distance_px", "") == "" else float(row["boundary_distance_px"])
        ),
        "signed_distance_px": (
            None if row.get("signed_distance_px", "") == "" else float(row["signed_distance_px"])
        ),
    }


def resolve_court_config(job: dict[str, Any], session: dict[str, Any]) -> tuple[Path | None, str | None, int]:
    path = job.get("court_config_path") or session.get("court_config_path")
    if not path:
        return None, None, 0
    config_path = Path(path)
    if not config_path.exists():
        raise RuntimeError(f"court config not found: {config_path}")
    config_image = job.get("config_image") or session.get("config_image")
    config_index = int(job.get("config_index", session.get("config_index", 0)) or 0)
    return config_path, config_image, config_index


def resolve_pressed_at_sec(
    source_path: Path,
    pressed_at_sec: float | None,
    use_video_end: bool,
    end_offset_sec: float,
) -> tuple[float, dict[str, Any]]:
    info = video_info(source_path)
    duration_sec = float(info.get("duration_sec") or 0.0)
    if pressed_at_sec is None or use_video_end:
        raw_pressed_at_sec = duration_sec - end_offset_sec
        source = "video_end"
    else:
        raw_pressed_at_sec = float(pressed_at_sec)
        source = "request"

    resolved = max(0.0, raw_pressed_at_sec)
    if duration_sec > 0:
        resolved = min(resolved, duration_sec)

    return resolved, {
        "source": source,
        "raw_pressed_at_sec": raw_pressed_at_sec,
        "pressed_at_sec": resolved,
        "video_duration_sec": duration_sec,
        "end_offset_sec": end_offset_sec,
        "clamped": resolved != raw_pressed_at_sec,
    }


def run_inout(
    current_job_dir: Path,
    clip_path: Path,
    track_csv: Path,
    bounce_csv: Path,
    court_config_path: Path,
    config_image: str | None,
    config_index: int,
    clip_start_sec: float,
    render_video: bool,
) -> dict[str, Any]:
    judged_csv = current_job_dir / "inout_judged.csv"
    combined_video = current_job_dir / "inout_overlay.avi"

    info = video_info(clip_path)
    config_args = SimpleNamespace(
        config_image=config_image,
        config_index=config_index,
        target_width=float(info["width"]),
        target_height=float(info["height"]),
        video_path=None,
    )
    config = normalize_config(load_json(court_config_path), config_args)
    judge_csv(
        input_csv=bounce_csv,
        output_csv=judged_csv,
        config=config,
        x_column="x",
        y_column="y",
    )
    rows = read_csv_rows(judged_csv)
    decisions = [inout_payload(row, clip_start_sec) for row in rows]

    artifacts = {"inout_csv": relative_file_url(judged_csv)}
    if render_video:
        overlay_args = SimpleNamespace(
            video_path=clip_path,
            court_config=court_config_path,
            judged_csv=judged_csv,
            output_path=combined_video,
            track_csv=track_csv,
            config_image=config_image,
            config_index=config_index,
            line_thickness=6,
            trace=8,
            bounce_display_window=4,
            hide_track=False,
        )
        write_overlay_video(overlay_args)
        if combined_video.exists():
            artifacts["inout_overlay_video"] = relative_file_url(combined_video)

    return {
        "status": "done" if decisions else "no_bounce",
        "court_config": {
            "path": str(court_config_path),
            "config_image": config_image,
            "config_index": config_index,
            "mode": config.get("mode"),
            "source_mode": config.get("source_mode"),
        },
        "decisions": decisions,
        "primary_decision": decisions[0] if decisions else None,
        "artifacts": artifacts,
    }


def run_preprocess(
    source_path: Path,
    current_job_dir: Path,
    pressed_at_sec: float,
    lookback_sec: float,
    render_video: bool,
    court_config_path: Path | None = None,
    config_image: str | None = None,
    config_index: int = 0,
    render_inout_video: bool = True,
) -> dict[str, Any]:
    from src.ball_tracking.infer_on_video import read_video
    from src.bounce_detection.detect_bounces import scale_track_to_frame, track_ball
    from src.bounce_detection.detect_bounces_from_track_csv import (
        detect_y_reversal_bounces,
        write_bounce_csv,
        write_track_csv,
        write_video as write_bounce_video,
    )

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

    inout = None
    if court_config_path is not None:
        inout = run_inout(
            current_job_dir=current_job_dir,
            clip_path=clip_path,
            track_csv=track_csv,
            bounce_csv=bounce_csv,
            court_config_path=court_config_path,
            config_image=config_image,
            config_index=config_index,
            clip_start_sec=start_sec,
            render_video=render_inout_video,
        )
        artifacts.update(inout["artifacts"])

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
            "court_config": None if court_config_path is None else str(court_config_path),
        },
        "inout": inout,
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
        pressed_at_sec, timing = resolve_pressed_at_sec(
            source_path=source_path,
            pressed_at_sec=job.get("pressed_at_sec"),
            use_video_end=bool(job.get("use_video_end", False)),
            end_offset_sec=float(job.get("end_offset_sec", 0.0)),
        )
        job["pressed_at_sec"] = pressed_at_sec
        job["timing"] = timing
        write_job(session_id, job)
        court_config_path, config_image, config_index = resolve_court_config(job, session)
        result = run_preprocess(
            source_path=source_path,
            current_job_dir=job_dir(session_id, job_id),
            pressed_at_sec=pressed_at_sec,
            lookback_sec=float(job["lookback_sec"]),
            render_video=bool(job.get("render_video", True)),
            court_config_path=court_config_path,
            config_image=config_image,
            config_index=config_index,
            render_inout_video=bool(job.get("render_inout_video", True)),
        )
        job.update({"status": "done", **result})
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
    write_job(session_id, job)


def build_frontend_job_result(job: dict[str, Any]) -> dict[str, Any]:
    inout = job.get("inout") or {}
    return {
        "job_id": job.get("job_id"),
        "session_id": job.get("session_id"),
        "status": job.get("status"),
        "result": job.get("result"),
        "confidence": job.get("confidence"),
        "timing": job.get("timing"),
        "primary_bounce": job.get("primary_bounce"),
        "primary_decision": inout.get("primary_decision"),
        "bounces": job.get("bounces") or [],
        "decisions": inout.get("decisions") or [],
        "artifacts": job.get("artifacts") or {},
        "error": job.get("error"),
    }


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def saved_response(record: dict[str, Any], include_job_result: bool) -> dict[str, Any]:
    response = {
        "id": record["id"],
        "session_id": record.get("session_id"),
        "job_id": record["job_id"],
        "created_at": record["created_at"],
        "recorded_at": record["recorded_at"],
        "recorded_date": record.get("recorded_date"),
        "match_type": record["match_type"],
        "decision": record["decision"],
        "decision_reason": record.get("decision_reason"),
        "video_path": record.get("video_path"),
        "video_url": record.get("video_url"),
        "clip_path": record.get("clip_path"),
        "clip_url": record.get("clip_url"),
        "result_video_url": record.get("result_video_url"),
        "inout_overlay_video_url": record.get("inout_overlay_video_url"),
        "inout_csv_url": record.get("inout_csv_url"),
        "confidence": record.get("confidence"),
        "primary_bounce": record.get("primary_bounce"),
        "primary_decision": record.get("primary_decision"),
    }
    if include_job_result:
        response["job_result"] = record["job_result"]
    return response


def build_saved_judgement_record(job: dict[str, Any], request: SaveJudgementRequest) -> dict[str, Any]:
    if job.get("status") != "done":
        raise HTTPException(status_code=409, detail="job is not done")

    result = build_frontend_job_result(job)
    primary_decision = result.get("primary_decision") or {}
    primary_bounce = result.get("primary_bounce")
    decision = primary_decision.get("decision") or "UNKNOWN"
    decision_reason = primary_decision.get("decision_reason")
    artifacts = result.get("artifacts") or {}

    session_id = job.get("session_id")
    session = read_session(session_id) if session_id else {}
    recording = session.get("recording") or {}
    video_path = session.get("recording_path")
    clip_url = artifacts.get("clip")
    clip_path = None
    if session_id:
        candidate = job_dir(session_id, job["job_id"]) / "clip.mp4"
        if candidate.exists():
            clip_path = str(candidate)

    created_at = now_utc_iso()
    if request.recorded_at:
        recorded_at = request.recorded_at.isoformat()
        recorded_date = request.recorded_date.isoformat() if request.recorded_date else recorded_at[:10]
    elif request.recorded_date:
        recorded_date = request.recorded_date.isoformat()
        recorded_at = f"{recorded_date}T00:00:00"
    else:
        recorded_at = created_at
        recorded_date = recorded_at[:10]
    return {
        "id": str(uuid.uuid4()),
        "session_id": session_id,
        "job_id": job["job_id"],
        "created_at": created_at,
        "recorded_at": recorded_at,
        "recorded_date": recorded_date,
        "match_type": request.match_type,
        "decision": decision,
        "decision_reason": decision_reason,
        "video_path": video_path,
        "video_url": recording.get("url"),
        "clip_path": clip_path,
        "clip_url": clip_url,
        "result_video_url": artifacts.get("result_video"),
        "inout_overlay_video_url": artifacts.get("inout_overlay_video"),
        "inout_csv_url": artifacts.get("inout_csv"),
        "confidence": result.get("confidence"),
        "primary_bounce_json": encode_json(primary_bounce),
        "primary_decision_json": encode_json(primary_decision),
        "job_result_json": encode_json(result),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return runtime_health()


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

    try:
        image = read_image_exif(frame_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail="could not read uploaded frame")
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


@app.post("/sessions/{session_id}/court-config/detect")
def detect_court_config(
    session_id: str,
    frame: UploadFile = File(...),
    family: str = "tag36h11",
    min_side_px: float = 0.0,
) -> dict[str, Any]:
    if family not in APRILTAG_FAMILIES:
        raise HTTPException(status_code=400, detail=f"unknown family: {family}")
    session = read_session(session_id)

    suffix = Path(frame.filename or "court_frame.jpg").suffix or ".jpg"
    frame_path = session_dir(session_id) / "court_config_frames" / f"{uuid.uuid4()}{suffix}"
    save_upload(frame, frame_path)

    try:
        record = detect_view2_court_config(frame_path, family, min_side_px)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail="could not read uploaded frame") from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"court line detection failed: {exc}") from exc

    config_path = session_dir(session_id) / "court_config_detected.json"
    court_config = [record]
    write_json(config_path, court_config)

    session["court_config_path"] = str(config_path)
    session["config_image"] = None
    session["config_index"] = 0
    session["court_config_detection"] = {
        "frame_path": str(frame_path),
        "frame_url": relative_file_url(frame_path),
        "court_config_path": str(config_path),
        "court_config_url": relative_file_url(config_path),
        "schema": record.get("schema"),
        "mode": record.get("mode"),
        "family": record.get("family"),
        "marker_count": record.get("marker_count"),
        "line_count": len(record.get("lines", [])),
    }
    write_session(session)

    return {
        "session_id": session_id,
        "court_config_path": str(config_path),
        "config_image": None,
        "config_index": 0,
        "url": relative_file_url(config_path),
        "frame": relative_file_url(frame_path),
        "court_config": court_config,
        "summary": session["court_config_detection"],
    }


@app.post("/sessions/{session_id}/court-config/upload")
def upload_court_config(
    session_id: str,
    config: UploadFile = File(...),
    config_image: str | None = None,
    config_index: int = 0,
) -> dict[str, Any]:
    session = read_session(session_id)
    suffix = Path(config.filename or "court_config.json").suffix or ".json"
    config_path = session_dir(session_id) / f"court_config{suffix}"
    save_upload(config, config_path)
    try:
        load_json(config_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid court config JSON: {exc}") from exc
    session["court_config_path"] = str(config_path)
    session["config_image"] = config_image
    session["config_index"] = config_index
    write_session(session)
    return {
        "session_id": session_id,
        "court_config_path": str(config_path),
        "config_image": config_image,
        "config_index": config_index,
        "url": relative_file_url(config_path),
    }


@app.post("/sessions/{session_id}/court-config/path")
def set_court_config_path(session_id: str, request: CourtConfigPathRequest) -> dict[str, Any]:
    session = read_session(session_id)
    config_path = Path(request.path)
    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"court config not found: {config_path}")
    try:
        load_json(config_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid court config JSON: {exc}") from exc
    session["court_config_path"] = str(config_path)
    session["config_image"] = request.config_image
    session["config_index"] = request.config_index
    write_session(session)
    return {
        "session_id": session_id,
        "court_config_path": str(config_path),
        "config_image": request.config_image,
        "config_index": request.config_index,
    }


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
        "use_video_end": request.use_video_end or request.pressed_at_sec is None,
        "end_offset_sec": request.end_offset_sec,
        "lookback_sec": request.lookback_sec,
        "render_video": request.render_video,
        "court_config_path": request.court_config_path,
        "config_image": request.config_image,
        "config_index": request.config_index,
        "render_inout_video": request.render_inout_video,
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
        "court_config_path": request.court_config_path,
        "config_image": request.config_image,
        "config_index": request.config_index,
    }
    write_session(session)

    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "session_id": session_id,
        "status": "pending",
        "pressed_at_sec": request.pressed_at_sec,
        "use_video_end": request.use_video_end or request.pressed_at_sec is None,
        "end_offset_sec": request.end_offset_sec,
        "lookback_sec": request.lookback_sec,
        "render_video": request.render_video,
        "court_config_path": request.court_config_path,
        "config_image": request.config_image,
        "config_index": request.config_index,
        "render_inout_video": request.render_inout_video,
    }
    write_job(session_id, job)
    background_tasks.add_task(run_judgement_job, session_id, job_id)
    return JudgeQueuedResponse(job_id=job_id, status="pending")


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return find_job(job_id)


@app.get("/jobs/{job_id}/result", response_model=FrontendJobResult)
def get_job_result(job_id: str) -> FrontendJobResult:
    return FrontendJobResult(**build_frontend_job_result(find_job(job_id)))


@app.post("/jobs/{job_id}/save", response_model=SavedJudgementDetail)
def save_job_result(job_id: str, request: SaveJudgementRequest) -> SavedJudgementDetail:
    job = find_job(job_id)
    record = insert_judgement_record(build_saved_judgement_record(job, request))
    return SavedJudgementDetail(**saved_response(record, include_job_result=True))


@app.get("/judgements", response_model=list[SavedJudgementResponse])
def list_saved_judgements(
    match_type: Literal["singles", "doubles"] | None = None,
    decision: str | None = None,
    recorded_date: date | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    limit: int = 50,
) -> list[SavedJudgementResponse]:
    limit = max(1, min(limit, 200))
    records = list_judgement_records(
        match_type=match_type,
        decision=decision,
        recorded_date=recorded_date.isoformat() if recorded_date else None,
        date_from=date_from.isoformat() if date_from else None,
        date_to=date_to.isoformat() if date_to else None,
        limit=limit,
    )
    return [
        SavedJudgementResponse(**saved_response(record, include_job_result=False))
        for record in records
    ]


@app.get("/judgements/{record_id}", response_model=SavedJudgementDetail)
def get_saved_judgement(record_id: str) -> SavedJudgementDetail:
    record = get_judgement_record(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown judgement id: {record_id}")
    return SavedJudgementDetail(**saved_response(record, include_job_result=True))


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
