
"""Phase 11.4G-12H-11A: conservative question constraint inventory.

Records observable surface signals in a question.

Does NOT:
- establish semantic interpretation;
- identify entity or title boundaries;
- prove constraint preservation;
- authorize decomposition or answer acceptance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ConstraintKind(str, Enum):
    NEGATION = "negation"
    TEMPORAL = "temporal"
    COMPARATIVE = "comparative"
    SUPERLATIVE = "superlative"
    UNCERTAINTY = "uncertainty"
    CONTEXTUAL = "contextual"


@dataclass(frozen=True)
class ConstraintSignal:
    kind: ConstraintKind
    text: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ConstraintKind):
            raise TypeError("kind must be ConstraintKind.")

        if not isinstance(self.text, str) or not self.text:
            raise ValueError("Signal text must be nonempty.")

        if (
            not isinstance(self.start, int)
            or not isinstance(self.end, int)
            or self.start < 0
            or self.end <= self.start
        ):
            raise ValueError("Invalid signal span.")


@dataclass(frozen=True)
class QuestionConstraintInventoryResult:
    question: str
    signals: tuple[ConstraintSignal, ...]

    @property
    def has_signals(self) -> bool:
        return bool(self.signals)


class QuestionConstraintInventory:
    name = "question_constraint_inventory_v1"

    _PATTERNS = (
        (
            ConstraintKind.NEGATION,
            re.compile(
                r"\b(?:not|never|neither|without)\b",
                re.IGNORECASE,
            ),
        ),
        (
            ConstraintKind.TEMPORAL,
            re.compile(
                r"\b(?:before|after|during|since|until|formerly)\b"
                r"|\b(?:18|19|20|21)\d{2}\b",
                re.IGNORECASE,
            ),
        ),
        (
            ConstraintKind.COMPARATIVE,
            re.compile(
                r"\b(?:older|younger|higher|lower|greater|"
                r"smaller|larger|taller|earlier|later)\b"
                r"|\bmore\s+than\b",
                re.IGNORECASE,
            ),
        ),
        (
            ConstraintKind.SUPERLATIVE,
            re.compile(
                r"\b(?:first|last|oldest|youngest|largest|"
                r"smallest|best|most)\b",
                re.IGNORECASE,
            ),
        ),
        (
            ConstraintKind.UNCERTAINTY,
            re.compile(
                r"\b(?:might|may|could|would|possibly|"
                r"perhaps|reportedly|alleged)\b",
                re.IGNORECASE,
            ),
        ),
        (
            ConstraintKind.CONTEXTUAL,
            re.compile(
                r"\b(?:in\s+the\s+film|in\s+the\s+movie|"
                r"in\s+the\s+book|in\s+the\s+series)\b",
                re.IGNORECASE,
            ),
        ),
    )

    def analyze(
        self,
        question: str,
    ) -> QuestionConstraintInventoryResult:
        if not isinstance(question, str):
            raise TypeError("question must be a string.")

        signals: list[ConstraintSignal] = []

        for kind, pattern in self._PATTERNS:
            for match in pattern.finditer(question):
                signals.append(
                    ConstraintSignal(
                        kind=kind,
                        text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                    )
                )

        signals.sort(
            key=lambda item: (
                item.start,
                item.end,
                item.kind.value,
            )
        )

        return QuestionConstraintInventoryResult(
            question=question,
            signals=tuple(signals),
        )
