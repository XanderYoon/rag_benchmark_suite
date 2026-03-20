from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


def append_run_telemetry(*, run_id: str, telemetry: dict, output_dir: Path) -> Path:
    """Append one benchmark telemetry record to a JSONL store and return its path."""
    normalized_run_id = str(run_id).strip()
    if not normalized_run_id:
        raise ValueError("run_id is required to append benchmark telemetry.")
    if not isinstance(telemetry, dict):
        raise ValueError("telemetry payload must be a dictionary.")

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    telemetry_file = output_path / "benchmark_telemetry.jsonl"

    record = {
        "run_id": normalized_run_id,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        **telemetry,
    }
    with telemetry_file.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(record, ensure_ascii=True))
        file_obj.write("\n")
    return telemetry_file


def append_debug_event(
    *,
    event_type: str,
    payload: dict,
    output_dir: Path | None = None,
) -> Path:
    """Append one structured benchmark debug event and return its JSONL path."""
    normalized_event_type = str(event_type).strip()
    if not normalized_event_type:
        raise ValueError("event_type is required to append a debug event.")
    if not isinstance(payload, dict):
        raise ValueError("debug payload must be a dictionary.")

    debug_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else Path("data/benchmark_runs/debug").resolve()
    )
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_file = debug_dir / "benchmark_debug.jsonl"
    record = {
        "event_type": normalized_event_type,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        **payload,
    }
    with debug_file.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(record, ensure_ascii=True))
        file_obj.write("\n")
    return debug_file


def load_recent_telemetry(*, output_dir: Path, limit: int = 50) -> list[dict]:
    """Load recent telemetry rows from disk, bounded by the requested limit."""
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}.")

    telemetry_file = Path(output_dir).expanduser().resolve() / "benchmark_telemetry.jsonl"
    if not telemetry_file.exists():
        return []

    rows: list[dict] = []
    with telemetry_file.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows[-limit:]
