
"""Phase 11.4G-12H-11I: conservative attachment comparison.

Compares explicitly supplied reference and proposed attachment sets.

Diagnostic only. Matching attachments do not establish semantic
fidelity unless the reference has been independently validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.recovery.constraint_attachment import (
    AttachedConstraint,
    ConstraintAttachmentSet,
)


class AttachmentComparisonStatus(str, Enum):
    MATCH = "match"
    MISSING = "missing"
    MODIFIED = "modified"
    EXTRA = "extra"


@dataclass(frozen=True)
class AttachmentComparison:
    constraint_id: str
    status: AttachmentComparisonStatus
    reference: AttachedConstraint | None
    proposed: AttachedConstraint | None
    reason: str


@dataclass(frozen=True)
class AttachmentComparisonResult:
    comparisons: tuple[AttachmentComparison, ...]

    @property
    def all_match(self) -> bool:
        return all(
            item.status == AttachmentComparisonStatus.MATCH
            for item in self.comparisons
        )


class ConstraintAttachmentComparator:
    name = "constraint_attachment_comparator_v1"

    def compare(
        self,
        reference: ConstraintAttachmentSet,
        proposed: ConstraintAttachmentSet,
    ) -> AttachmentComparisonResult:
        if not isinstance(reference, ConstraintAttachmentSet):
            raise TypeError("reference must be ConstraintAttachmentSet")

        if not isinstance(proposed, ConstraintAttachmentSet):
            raise TypeError("proposed must be ConstraintAttachmentSet")

        if reference.requirements != proposed.requirements:
            raise ValueError(
                "Attachment sets must share identical requirements."
            )

        reference_by_id = {
            item.constraint_id: item
            for item in reference.constraints
        }
        proposed_by_id = {
            item.constraint_id: item
            for item in proposed.constraints
        }

        comparisons = []

        for constraint_id, original in reference_by_id.items():
            candidate = proposed_by_id.get(constraint_id)

            if candidate is None:
                status = AttachmentComparisonStatus.MISSING
                reason = "reference_constraint_absent"

            elif candidate == original:
                status = AttachmentComparisonStatus.MATCH
                reason = "exact_attachment_match"

            else:
                status = AttachmentComparisonStatus.MODIFIED
                reason = "attachment_fields_differ"

            comparisons.append(
                AttachmentComparison(
                    constraint_id=constraint_id,
                    status=status,
                    reference=original,
                    proposed=candidate,
                    reason=reason,
                )
            )

        for constraint_id, candidate in proposed_by_id.items():
            if constraint_id in reference_by_id:
                continue

            comparisons.append(
                AttachmentComparison(
                    constraint_id=constraint_id,
                    status=AttachmentComparisonStatus.EXTRA,
                    reference=None,
                    proposed=candidate,
                    reason="unexpected_proposed_constraint",
                )
            )

        return AttachmentComparisonResult(
            comparisons=tuple(comparisons)
        )
