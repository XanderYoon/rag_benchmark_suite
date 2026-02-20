from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class AuditLog:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def append(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append({
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        })
