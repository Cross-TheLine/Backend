from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


DB_PATH = Path("output") / "judgements.sqlite3"


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS judgement_records (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                job_id TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                recorded_date TEXT,
                match_type TEXT NOT NULL CHECK (match_type IN ('singles', 'doubles')),
                decision TEXT NOT NULL,
                decision_reason TEXT,
                video_path TEXT,
                video_url TEXT,
                clip_path TEXT,
                clip_url TEXT,
                result_video_url TEXT,
                inout_overlay_video_url TEXT,
                inout_csv_url TEXT,
                confidence REAL,
                primary_bounce_json TEXT,
                primary_decision_json TEXT,
                job_result_json TEXT NOT NULL
            )
            """
        )
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(judgement_records)").fetchall()
        }
        if "recorded_date" not in columns:
            conn.execute("ALTER TABLE judgement_records ADD COLUMN recorded_date TEXT")
            conn.execute(
                "UPDATE judgement_records SET recorded_date = substr(recorded_at, 1, 10) "
                "WHERE recorded_date IS NULL"
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_judgement_records_created_at "
            "ON judgement_records(created_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_judgement_records_recorded_date "
            "ON judgement_records(recorded_date DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_judgement_records_match_type "
            "ON judgement_records(match_type)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_judgement_records_decision "
            "ON judgement_records(decision)"
        )


def encode_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def decode_json(value: str | None) -> Any:
    if value is None:
        return None
    return json.loads(value)


def row_to_record(row: sqlite3.Row) -> dict[str, Any]:
    record = dict(row)
    record["primary_bounce"] = decode_json(record.pop("primary_bounce_json"))
    record["primary_decision"] = decode_json(record.pop("primary_decision_json"))
    record["job_result"] = decode_json(record.pop("job_result_json"))
    return record


def insert_judgement_record(record: dict[str, Any]) -> dict[str, Any]:
    columns = [
        "id",
        "session_id",
        "job_id",
        "created_at",
        "recorded_at",
        "recorded_date",
        "match_type",
        "decision",
        "decision_reason",
        "video_path",
        "video_url",
        "clip_path",
        "clip_url",
        "result_video_url",
        "inout_overlay_video_url",
        "inout_csv_url",
        "confidence",
        "primary_bounce_json",
        "primary_decision_json",
        "job_result_json",
    ]
    placeholders = ", ".join("?" for _ in columns)
    update_columns = [column for column in columns if column not in {"id", "job_id", "created_at"}]
    update_sql = ", ".join(f"{column} = excluded.{column}" for column in update_columns)
    with connect() as conn:
        conn.execute(
            f"""
            INSERT INTO judgement_records ({', '.join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(job_id) DO UPDATE SET {update_sql}
            """,
            [record.get(column) for column in columns],
        )
        row = conn.execute(
            "SELECT * FROM judgement_records WHERE job_id = ?",
            (record["job_id"],),
        ).fetchone()
    return row_to_record(row)


def list_judgement_records(
    match_type: str | None = None,
    decision: str | None = None,
    recorded_date: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if match_type:
        clauses.append("match_type = ?")
        params.append(match_type)
    if decision:
        clauses.append("decision = ?")
        params.append(decision)
    if recorded_date:
        clauses.append("recorded_date = ?")
        params.append(recorded_date)
    if date_from:
        clauses.append("recorded_date >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("recorded_date <= ?")
        params.append(date_to)

    where_sql = "" if not clauses else "WHERE " + " AND ".join(clauses)
    params.append(limit)
    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM judgement_records
            {where_sql}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [row_to_record(row) for row in rows]


def get_judgement_record(record_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM judgement_records WHERE id = ?",
            (record_id,),
        ).fetchone()
    if row is None:
        return None
    return row_to_record(row)
