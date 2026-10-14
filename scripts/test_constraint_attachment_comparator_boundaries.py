
"""Phase 11.4G-12H-11J: attachment comparator boundary tests.

Documents limitations of ID-based comparison, identical-graph
requirements, and empty-set agreement.

Diagnostic only; no semantic acceptance.
"""

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


def requirements(
    predicate: str = "held_position",
) -> QuestionRequirements:
    return QuestionRequirements(
        question="Who held the position before 1998?",
        relationships=(
            RequiredRelationship(
                relationship_id="r1",
                subject="?person",
                predicate=predicate,
                object="?position",
            ),
        ),
        answer_variable="?person",
    )


def constraint(
    constraint_id: str,
) -> AttachedConstraint:
    return AttachedConstraint(
        constraint_id=constraint_id,
        kind=AttachmentConstraintKind.TEMPORAL,
        operator="before",
        value="1998",
        relationship_id="r1",
    )


def main() -> None:
    comparator = ConstraintAttachmentComparator()
    req = requirements()

    # J01: Same constraint fields, different IDs.
    reference = ConstraintAttachmentSet(
        requirements=req,
        constraints=(constraint("c1"),),
    )
    proposed = ConstraintAttachmentSet(
        requirements=req,
        constraints=(constraint("c99"),),
    )

    result = comparator.compare(reference, proposed)

    statuses = tuple(
        item.status for item in result.comparisons
    )

    assert statuses == (
        AttachmentComparisonStatus.MISSING,
        AttachmentComparisonStatus.EXTRA,
    )
    assert not result.all_match
    print("PASS J01_changed_constraint_id")

    # J02: Different relationship graphs are rejected.
    different_graph = ConstraintAttachmentSet(
        requirements=requirements("worked_at"),
        constraints=(constraint("c1"),),
    )

    try:
        comparator.compare(reference, different_graph)
    except ValueError:
        print("PASS J02_different_relationship_graph")
    else:
        raise AssertionError(
            "J02: expected ValueError for different requirements"
        )

    # J03: Two empty sets produce no comparisons.
    empty = ConstraintAttachmentSet(
        requirements=req,
        constraints=(),
    )

    result = comparator.compare(empty, empty)

    assert result.comparisons == ()
    assert result.all_match

    print(
        "PASS J03_both_empty: all_match=True "
        "(vacuous agreement, not semantic verification)"
    )

    # J04: Empty reference, nonempty proposal.
    result = comparator.compare(empty, proposed)

    assert tuple(
        item.status for item in result.comparisons
    ) == (AttachmentComparisonStatus.EXTRA,)
    assert not result.all_match

    print("PASS J04_empty_reference_extra_proposal")

    # J05: Nonempty reference, empty proposal.
    result = comparator.compare(reference, empty)

    assert tuple(
        item.status for item in result.comparisons
    ) == (AttachmentComparisonStatus.MISSING,)
    assert not result.all_match

    print("PASS J05_nonempty_reference_empty_proposal")

    print("\nComparator boundary tests: 5/5")


if __name__ == "__main__":
    main()
