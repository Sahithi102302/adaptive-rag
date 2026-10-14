
"""Tests for Phase 11.4G-12H-11I attachment comparator."""

from __future__ import annotations

from src.recovery.question_requirements import (
    QuestionRequirements,
    RequiredRelationship,
)
from src.recovery.constraint_attachment import (
    AttachedConstraint,
    AttachmentConstraintKind,
    ConstraintAttachmentSet,
)
from src.recovery.constraint_attachment_comparator import (
    AttachmentComparisonStatus,
    ConstraintAttachmentComparator,
)


def make_requirements() -> QuestionRequirements:
    return QuestionRequirements(
        question="Who held the position before 1998?",
        relationships=(
            RequiredRelationship(
                relationship_id="r1",
                subject="?person",
                predicate="held_position",
                object="?position",
            ),
            RequiredRelationship(
                relationship_id="r2",
                subject="?person",
                predicate="worked_at",
                object="?organization",
            ),
        ),
        answer_variable="?person",
    )


def make_constraint(
    constraint_id: str = "c1",
    operator: str = "before",
    value: str = "1998",
    relationship_id: str = "r1",
) -> AttachedConstraint:
    return AttachedConstraint(
        constraint_id=constraint_id,
        kind=AttachmentConstraintKind.TEMPORAL,
        operator=operator,
        value=value,
        relationship_id=relationship_id,
    )


def main() -> None:
    requirements = make_requirements()
    comparator = ConstraintAttachmentComparator()

    reference = ConstraintAttachmentSet(
        requirements=requirements,
        constraints=(make_constraint(),),
    )

    cases = (
        (
            "I01_exact_match",
            (make_constraint(),),
            (AttachmentComparisonStatus.MATCH,),
        ),
        (
            "I02_missing",
            (),
            (AttachmentComparisonStatus.MISSING,),
        ),
        (
            "I03_reversed_operator",
            (make_constraint(operator="after"),),
            (AttachmentComparisonStatus.MODIFIED,),
        ),
        (
            "I04_wrong_relationship",
            (make_constraint(relationship_id="r2"),),
            (AttachmentComparisonStatus.MODIFIED,),
        ),
        (
            "I05_extra_constraint",
            (
                make_constraint(),
                make_constraint(
                    constraint_id="c2",
                    operator="after",
                ),
            ),
            (
                AttachmentComparisonStatus.MATCH,
                AttachmentComparisonStatus.EXTRA,
            ),
        ),
    )

    for case_id, constraints, expected in cases:
        proposed = ConstraintAttachmentSet(
            requirements=requirements,
            constraints=constraints,
        )

        result = comparator.compare(reference, proposed)

        actual = tuple(
            item.status for item in result.comparisons
        )

        assert actual == expected, (
            f"{case_id}: expected {expected}, got {actual}"
        )

        assert result.all_match == all(
            status == AttachmentComparisonStatus.MATCH
            for status in expected
        )

        print(
            f"PASS {case_id}: "
            f"{[status.value for status in actual]}"
        )

    print(
        f"\nAttachment comparator tests: "
        f"{len(cases)}/{len(cases)}"
    )


if __name__ == "__main__":
    main()
