
"""Phase 11.4G-12H-11K: conservative comparison interpretation.

Interprets exact-field attachment comparisons.

Does not verify semantic fidelity, reference correctness,
constraint extraction completeness, or evidence grounding.
Never authorizes decomposition acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.recovery.constraint_attachment import (
    ConstraintAttachmentSet,
)
from src.recovery.constraint_attachment_comparator import (
    AttachmentComparisonResult,
    AttachmentComparisonStatus,
    ConstraintAttachmentComparator,
)


class ComparisonInterpretationStatus(str, Enum):
    AGREEMENT_OBSERVED = "agreement_observed"
    DISAGREEMENT_OBSERVED = "disagreement_observed"
    INSUFFICIENT_REFERENCE = "insufficient_reference"


@dataclass(frozen=True)
class ComparisonInterpretationResult:
    status: ComparisonInterpretationStatus
    comparison: AttachmentComparisonResult
    reason: str


class ConstraintComparisonInterpreter:
    name = "constraint_comparison_interpreter_v1"

    def __init__(self) -> None:
        self.comparator = ConstraintAttachmentComparator()

    def interpret(
        self,
        reference: ConstraintAttachmentSet,
        proposed: ConstraintAttachmentSet,
    ) -> ComparisonInterpretationResult:
        comparison = self.comparator.compare(
            reference,
            proposed,
        )

        # An empty reference cannot establish coverage.
        # This is intentionally checked before all_match.
        if not reference.constraints:
            return ComparisonInterpretationResult(
                status=(
                    ComparisonInterpretationStatus
                    .INSUFFICIENT_REFERENCE
                ),
                comparison=comparison,
                reason="reference_contains_no_constraints",
            )

        if any(
            item.status != AttachmentComparisonStatus.MATCH
            for item in comparison.comparisons
        ):
            return ComparisonInterpretationResult(
                status=(
                    ComparisonInterpretationStatus
                    .DISAGREEMENT_OBSERVED
                ),
                comparison=comparison,
                reason="attachment_comparison_difference",
            )

        return ComparisonInterpretationResult(
            status=(
                ComparisonInterpretationStatus
                .AGREEMENT_OBSERVED
            ),
            comparison=comparison,
            reason="nonempty_exact_field_agreement",
        )
