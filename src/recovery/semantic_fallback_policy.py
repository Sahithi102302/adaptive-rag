
"""Phase 11.4G-12F: conservative semantic fallback screening.

Screens out-of-grammar questions for obvious risk indicators.

Important:
- This is a heuristic screening layer, not semantic verification.
- CANDIDATE means eligible for further review, never acceptance.
- A question without detected risk indicators may still be unsafe.
- Title-boundary handling is partial and must not be treated as
  a general solution to entity extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class FallbackEligibility(str, Enum):
    CANDIDATE = "candidate"
    BLOCKED = "blocked"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class FallbackScreeningResult:
    eligibility: FallbackEligibility
    reason: str

    @property
    def eligible_for_review(self) -> bool:
        return self.eligibility == FallbackEligibility.CANDIDATE


class SemanticFallbackPolicy:
    name = "semantic_fallback_policy_v3"

    _BLOCKING_PATTERNS = (
        (
            "negation",
            re.compile(
                r"\b(?:not|never|neither|without)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "uncertainty_or_hypothetical",
            re.compile(
                r"\b(?:might|may|could|would|supposed|"
                r"alleged|reportedly|possibly|perhaps)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "unsupported_role",
            re.compile(
                r"\b(?:co-author|co-writer|ghostwriter|"
                r"narrator|editor|translator|illustrator|"
                r"publisher|fictional author|authors)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "temporal_constraint",
            re.compile(
                r"\b(?:in|during|since|until|by|around)\s+"
                r"(?:18|19|20|21)\d{2}\b",
                re.IGNORECASE,
            ),
        ),
        (
            "additional_constraint",
            re.compile(
                r"\b(?:first|before|after|during|"
                r"while|later|earlier|alongside|with their)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "compound_request",
            re.compile(
                r",\s*and\b|\band what\b|\band who\b|"
                r"\band when\b|\band where\b",
                re.IGNORECASE,
            ),
        ),
        (
            "unsupported_relation",
            re.compile(
                r"\b(?:employed|worked|taught|hired|"
                r"awarded|degree|graduated|applied|"
                r"affiliated|appointed)\b",
                re.IGNORECASE,
            ),
        ),
    )

    # Handle "last" separately because it frequently appears
    # in book titles, such as "The Last Meridian".
    #
    # Only explicit trailing constraints are detected here.
    _TRAILING_LAST_CONSTRAINT = re.compile(
        r"\b(?:attend|attended|study|studied)\s+last\s*\?$",
        re.IGNORECASE,
    )

    _POTENTIAL_EDUCATION_CUES = re.compile(
        r"\b(?:university|college|attend|attended|"
        r"study|studied|educated|education)\b",
        re.IGNORECASE,
    )

    _POTENTIAL_AUTHORSHIP_CUES = re.compile(
        r"\b(?:author|writer|novelist|authored|wrote)\b",
        re.IGNORECASE,
    )

    def screen(self, question: str) -> FallbackScreeningResult:
        if not isinstance(question, str) or not question.strip():
            return FallbackScreeningResult(
                eligibility=FallbackEligibility.BLOCKED,
                reason="empty_question",
            )

        normalized = " ".join(question.split())

        # Step 1: Detect explicit trailing "last" constraints.
        if self._TRAILING_LAST_CONSTRAINT.search(normalized):
            return FallbackScreeningResult(
                eligibility=FallbackEligibility.BLOCKED,
                reason="additional_constraint",
            )

        # Step 2: Detect unsupported semantic operations.
        for reason, pattern in self._BLOCKING_PATTERNS:
            if pattern.search(normalized):
                return FallbackScreeningResult(
                    eligibility=FallbackEligibility.BLOCKED,
                    reason=reason,
                )

        # Step 3: Check for basic task-related cues.
        if (
            not self._POTENTIAL_EDUCATION_CUES.search(normalized)
            or not self._POTENTIAL_AUTHORSHIP_CUES.search(normalized)
        ):
            return FallbackScreeningResult(
                eligibility=FallbackEligibility.UNRESOLVED,
                reason="missing_task_cues",
            )

        # Step 4: Mark for further verification only.
        return FallbackScreeningResult(
            eligibility=FallbackEligibility.CANDIDATE,
            reason="potential_paraphrase_for_review_only",
        )
