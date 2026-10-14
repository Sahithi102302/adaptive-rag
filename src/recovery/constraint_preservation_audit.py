
"""Phase 11.4G-12H-11E: lexical constraint preservation audit.

Compares original surface signals against proposed subquestion text.

PRESENT means lexical occurrence only, not semantic preservation.
MISSING means the lexical signal was not found, not that the
semantic constraint was necessarily lost.

Diagnostic only. Never authorizes decomposition acceptance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from src.recovery.question_constraint_inventory import (
    ConstraintSignal,
    QuestionConstraintInventory,
)
from src.recovery.constraint_signal_review import (
    ConstraintSignalReviewer,
    SignalReviewStatus,
)


class LexicalStatus(str, Enum):
    PRESENT = "present"
    MISSING = "missing"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class ConstraintLexicalObservation:
    signal: ConstraintSignal
    status: LexicalStatus
    reason: str


@dataclass(frozen=True)
class ConstraintPreservationAuditResult:
    question: str
    subquestions: tuple[str, ...]
    observations: tuple[ConstraintLexicalObservation, ...]

    @property
    def has_missing_signals(self) -> bool:
        return any(
            item.status == LexicalStatus.MISSING
            for item in self.observations
        )


class ConstraintPreservationAuditor:
    name = "constraint_preservation_auditor_v1"

    def __init__(self) -> None:
        self.inventory = QuestionConstraintInventory()
        self.reviewer = ConstraintSignalReviewer()

    @staticmethod
    def _contains_signal(text: str, signal: str) -> bool:
        pattern = re.compile(
            r"(?<!\w)" + re.escape(signal) + r"(?!\w)",
            re.IGNORECASE,
        )
        return bool(pattern.search(text))

    def audit(
        self,
        question: str,
        subquestions: tuple[str, ...],
    ) -> ConstraintPreservationAuditResult:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be nonempty.")

        if (
            not isinstance(subquestions, tuple)
            or not subquestions
            or not all(
                isinstance(item, str) and item.strip()
                for item in subquestions
            )
        ):
            raise ValueError(
                "subquestions must be a nonempty tuple "
                "of nonempty strings."
            )

        inventory = self.inventory.analyze(question)
        reviewed = self.reviewer.review(inventory)

        observations = []

        for item in reviewed.reviewed_signals:
            signal = item.signal

            if item.status == SignalReviewStatus.POTENTIALLY_AMBIGUOUS:
                status = LexicalStatus.UNRESOLVED
                reason = "ambiguous_original_signal"

            elif any(
                self._contains_signal(subquestion, signal.text)
                for subquestion in subquestions
            ):
                status = LexicalStatus.PRESENT
                reason = "lexical_signal_found"

            else:
                status = LexicalStatus.MISSING
                reason = "lexical_signal_absent"

            observations.append(
                ConstraintLexicalObservation(
                    signal=signal,
                    status=status,
                    reason=reason,
                )
            )

        return ConstraintPreservationAuditResult(
            question=question,
            subquestions=subquestions,
            observations=tuple(observations),
        )
