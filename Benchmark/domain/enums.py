from __future__ import annotations

from enum import Enum


class QuestionStatus(str, Enum):
    DRAFT = "draft"
    VERIFIED = "verified"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class DifficultyLabel(str, Enum):
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    DEFINITION = "definition"
    COMPARISON = "comparison"
    NEGATIVE = "negative"
