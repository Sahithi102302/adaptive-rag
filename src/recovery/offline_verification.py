
"""Offline evaluation of post-recovery evidence.

Gold annotations are used ONLY for evaluation.
Never import this module into the runtime verification pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from src.recovery.evidence_assessment import (
    EvidenceAssessmentBaseline,
    EvidenceComparison,
)


@dataclass(frozen=True)
class OfflineVerificationOutcome:
    gold_total: int
    original_gold_count: int
    candidate_gold_count: int
    recovered_count: int
    lost_count: int
    originally_complete: bool
    candidate_complete: bool
    repaired: bool
    harmed: bool
    complete_to_incomplete: bool
    comparison: EvidenceComparison


def evaluate_offline_verification(
    query: str,
    original_results: Sequence[Any],
    candidate_results: Sequence[Any],
    gold_evidence: set,
    evidence_extractor: Callable[[Sequence[Any]], set],
    assessor: EvidenceAssessmentBaseline | None = None,
) -> OfflineVerificationOutcome:
    """Evaluate a fixed candidate against gold supporting facts.

    The assessment itself is computed without gold evidence.
    """
    if not gold_evidence:
        raise ValueError("Offline evaluation requires nonempty gold evidence.")

    if not callable(evidence_extractor):
        raise TypeError("evidence_extractor must be callable.")

    baseline = assessor or EvidenceAssessmentBaseline()

    # Runtime-compatible signals are computed independently of gold.
    comparison = baseline.compare(
        query=query,
        original_results=original_results,
        candidate_results=candidate_results,
    )

    # Oracle-only evaluation begins here.
    original_gold = set(evidence_extractor(original_results)) & gold_evidence
    candidate_gold = set(evidence_extractor(candidate_results)) & gold_evidence

    recovered = candidate_gold - original_gold
    lost = original_gold - candidate_gold

    originally_complete = gold_evidence.issubset(original_gold)
    candidate_complete = gold_evidence.issubset(candidate_gold)

    return OfflineVerificationOutcome(
        gold_total=len(gold_evidence),
        original_gold_count=len(original_gold),
        candidate_gold_count=len(candidate_gold),
        recovered_count=len(recovered),
        lost_count=len(lost),
        originally_complete=originally_complete,
        candidate_complete=candidate_complete,
        repaired=not originally_complete and candidate_complete,
        harmed=bool(lost),
        complete_to_incomplete=(
            originally_complete and not candidate_complete
        ),
        comparison=comparison,
    )
