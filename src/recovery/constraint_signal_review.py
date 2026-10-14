
"""Phase 11.4G-12H-11C: conservative constraint signal review.

Annotates surface signals with potential ambiguity indicators.

Does not prove semantic meaning, suppress original signals,
or authorize decomposition acceptance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from src.recovery.question_constraint_inventory import (
    ConstraintKind,
    ConstraintSignal,
    QuestionConstraintInventoryResult,
)


class SignalReviewStatus(str, Enum):
    NEEDS_REVIEW = "needs_review"
    POTENTIALLY_AMBIGUOUS = "potentially_ambiguous"


@dataclass(frozen=True)
class ReviewedConstraintSignal:
    signal: ConstraintSignal
    status: SignalReviewStatus
    reason: str


@dataclass(frozen=True)
class ConstraintSignalReviewResult:
    question: str
    reviewed_signals: tuple[ReviewedConstraintSignal, ...]


class ConstraintSignalReviewer:
    name = "constraint_signal_reviewer_v1"

    _MONTH_MAY = re.compile(
        r"\bMay\s+\d{1,2}\s*,?\s*(?:18|19|20|21)\d{2}\b",
        re.IGNORECASE,
    )

    _GREATER_PLACE = re.compile(
        r"\bGreater\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
    )

    _OLDER_RELATION = re.compile(
        r"\bolder\s+(?:brother|sister|son|daughter)\b",
        re.IGNORECASE,
    )

    def review(
        self,
        inventory: QuestionConstraintInventoryResult,
    ) -> ConstraintSignalReviewResult:
        question = inventory.question

        reviewed: list[ReviewedConstraintSignal] = []

        for signal in inventory.signals:
            status = SignalReviewStatus.NEEDS_REVIEW
            reason = "surface_signal_requires_interpretation"

            ambiguity_patterns = ()

            if (
                signal.kind == ConstraintKind.UNCERTAINTY
                and signal.text.lower() == "may"
            ):
                ambiguity_patterns = (
                    ("possible_calendar_month", self._MONTH_MAY),
                )

            elif (
                signal.kind == ConstraintKind.COMPARATIVE
                and signal.text.lower() == "greater"
            ):
                ambiguity_patterns = (
                    ("possible_geographic_name", self._GREATER_PLACE),
                )

            elif (
                signal.kind == ConstraintKind.COMPARATIVE
                and signal.text.lower() == "older"
            ):
                ambiguity_patterns = (
                    ("possible_family_relation_modifier", self._OLDER_RELATION),
                )

            for ambiguity_reason, pattern in ambiguity_patterns:
                if any(
                    match.start() <= signal.start
                    and signal.end <= match.end()
                    for match in pattern.finditer(question)
                ):
                    status = SignalReviewStatus.POTENTIALLY_AMBIGUOUS
                    reason = ambiguity_reason
                    break

            reviewed.append(
                ReviewedConstraintSignal(
                    signal=signal,
                    status=status,
                    reason=reason,
                )
            )

        return ConstraintSignalReviewResult(
            question=question,
            reviewed_signals=tuple(reviewed),
        )
