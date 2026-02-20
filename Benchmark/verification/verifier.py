from __future__ import annotations

from datetime import datetime, timezone

from Benchmark.domain.enums import QuestionStatus
from Benchmark.domain.models import BenchmarkRecord


class Verifier:
    def verify(self, record: BenchmarkRecord, verified_by: str, notes: str = "") -> None:
        record.status = QuestionStatus.VERIFIED
        record.audit = {
            "verified_by": verified_by,
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "verification_notes": notes,
        }
        record.touch()

    def reject(self, record: BenchmarkRecord, verified_by: str, notes: str = "") -> None:
        record.status = QuestionStatus.REJECTED
        record.audit = {
            "verified_by": verified_by,
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "verification_notes": notes,
        }
        record.touch()

    def needs_revision(self, record: BenchmarkRecord, verified_by: str, notes: str = "") -> None:
        record.status = QuestionStatus.NEEDS_REVISION
        record.audit = {
            "verified_by": verified_by,
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "verification_notes": notes,
        }
        record.touch()
