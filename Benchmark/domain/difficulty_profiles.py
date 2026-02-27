from __future__ import annotations

from Benchmark.domain.enums import DifficultyLabel


DIFFICULTY_PROFILES: list[tuple[str, str, DifficultyLabel]] = [
    ("Single document: single hop", "single_single", DifficultyLabel.SINGLE_HOP),
    ("Single document: multi hop", "single_multi", DifficultyLabel.MULTI_HOP),
    ("Multiple documents", "multiple", DifficultyLabel.DEFINITION),
    ("Comparison", "comparison", DifficultyLabel.COMPARISON),
    ("Negative / Null", "negative", DifficultyLabel.NEGATIVE),
]

LEGACY_PROFILE_LABELS: dict[str, str] = {
    "direct reference: single hop": "Single document: single hop",
    "direct reference: multi hop": "Single document: multi hop",
    "single reference: single hop": "Single document: single hop",
    "single reference: multi hop": "Single document: multi hop",
    "multiple reference: multi hop": "Multiple documents",
}


def difficulty_profile_labels() -> list[str]:
    """Return the supported UI labels for question difficulty profiles."""
    return [label for label, _, _ in DIFFICULTY_PROFILES]


def canonical_profile_label(profile_label: str | None) -> str:
    """Normalize legacy and case-variant profile labels to the canonical UI label."""
    if profile_label is None:
        return DIFFICULTY_PROFILES[0][0]

    normalized = profile_label.strip()
    if not normalized:
        return DIFFICULTY_PROFILES[0][0]

    lowered = normalized.lower()
    for label in difficulty_profile_labels():
        if lowered == label.lower():
            return label

    if lowered in LEGACY_PROFILE_LABELS:
        return LEGACY_PROFILE_LABELS[lowered]

    for legacy_label, current_label in LEGACY_PROFILE_LABELS.items():
        if lowered == legacy_label.lower():
            return current_label

    return DIFFICULTY_PROFILES[0][0]


def difficulty_from_profile_label(profile_label: str | None) -> DifficultyLabel:
    """Map a profile label to the internal difficulty enum."""
    canonical_label = canonical_profile_label(profile_label)
    for label, _, difficulty in DIFFICULTY_PROFILES:
        if label == canonical_label:
            return difficulty
    return DIFFICULTY_PROFILES[0][2]


def reference_type_from_profile_label(profile_label: str | None) -> str:
    """Map a profile label to the generation reference type key."""
    canonical_label = canonical_profile_label(profile_label)
    for label, reference_type, _ in DIFFICULTY_PROFILES:
        if label == canonical_label:
            return reference_type
    return DIFFICULTY_PROFILES[0][1]

