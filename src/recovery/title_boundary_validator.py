
"""Phase 11.4G-12H-9: evidence-guided title disambiguation.

Compares all grammar-supported candidate title boundaries against
independently supplied document-title metadata.

A title match supports entity identification only.
It does not establish semantic equivalence, factual grounding,
or permission to accept a decomposition.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from src.recovery.title_span_extractor import (
    ConservativeTitleSpanExtractor,
    TitleBoundaryCandidate,
)


class BoundaryStatus(str, Enum):
    VERIFIED_TITLE = "verified_title"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class BoundaryValidationResult:
    status: BoundaryStatus
    title: str | None
    start: int | None
    end: int | None
    reason: str

    @property
    def verified(self) -> bool:
        return self.status == BoundaryStatus.VERIFIED_TITLE


class EvidenceBackedTitleBoundaryValidator:
    name = "evidence_backed_title_boundary_validator_v2"

    def __init__(self) -> None:
        self.extractor = ConservativeTitleSpanExtractor()

    @staticmethod
    def _normalize_title(title: str) -> str:
        return re.sub(r"\s+", " ", title).strip().casefold()

    def validate(
        self,
        question: str,
        known_titles: Iterable[str],
    ) -> BoundaryValidationResult:
        if not isinstance(question, str) or not question.strip():
            return self._unresolved("empty_question")

        title_keys = {
            self._normalize_title(title)
            for title in known_titles
            if isinstance(title, str) and title.strip()
        }

        if not title_keys:
            return self._unresolved("no_title_evidence")

        candidates: tuple[TitleBoundaryCandidate, ...] = (
            self.extractor.enumerate_candidates(question)
        )

        if not candidates:
            return self._unresolved("no_boundary_candidates")

        matched = [
            candidate
            for candidate in candidates
            if self._normalize_title(candidate.title) in title_keys
        ]

        if not matched:
            return self._unresolved(
                "no_candidate_matches_known_titles"
            )

        unique_spans = {
            (candidate.start, candidate.end)
            for candidate in matched
        }

        if len(unique_spans) != 1:
            return self._unresolved(
                "multiple_evidence_supported_boundaries"
            )

        candidate = matched[0]

        if question[candidate.start:candidate.end] != candidate.title:
            return self._unresolved("span_mismatch")

        return BoundaryValidationResult(
            status=BoundaryStatus.VERIFIED_TITLE,
            title=candidate.title,
            start=candidate.start,
            end=candidate.end,
            reason="unique_evidence_supported_title_boundary",
        )

    @staticmethod
    def _unresolved(reason: str) -> BoundaryValidationResult:
        return BoundaryValidationResult(
            status=BoundaryStatus.UNRESOLVED,
            title=None,
            start=None,
            end=None,
            reason=reason,
        )
