
"""Tests for Phase 11.4G-12H-11G constraint attachment contracts."""

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
        ),
        answer_variable="?person",
    )


def make_constraint(
    constraint_id: str = "c1",
    relationship_id: str = "r1",
) -> AttachedConstraint:
    return AttachedConstraint(
        constraint_id=constraint_id,
        kind=AttachmentConstraintKind.TEMPORAL,
        operator="before",
        value="1998",
        relationship_id=relationship_id,
    )


def expect_value_error(test_id: str, operation) -> None:
    try:
        operation()
    except ValueError:
        print(f"PASS {test_id}")
        return

    raise AssertionError(
        f"{test_id}: expected ValueError"
    )


def main() -> None:
    requirements = make_requirements()

    valid = ConstraintAttachmentSet(
        requirements=requirements,
        constraints=(make_constraint(),),
    )

    assert valid.has_constraints
    print("PASS T01_valid_attachment")

    empty = ConstraintAttachmentSet(
        requirements=requirements,
        constraints=(),
    )

    assert not empty.has_constraints
    print("PASS T02_empty_attachment_set")

    expect_value_error(
        "T03_unknown_relationship",
        lambda: ConstraintAttachmentSet(
            requirements=requirements,
            constraints=(
                make_constraint(relationship_id="r99"),
            ),
        ),
    )

    expect_value_error(
        "T04_duplicate_constraint_id",
        lambda: ConstraintAttachmentSet(
            requirements=requirements,
            constraints=(
                make_constraint(),
                make_constraint(),
            ),
        ),
    )

    expect_value_error(
        "T05_empty_operator",
        lambda: AttachedConstraint(
            constraint_id="c1",
            kind=AttachmentConstraintKind.TEMPORAL,
            operator="",
            value="1998",
            relationship_id="r1",
        ),
    )

    print("\nConstraint attachment tests: 5/5")


if __name__ == "__main__":
    main()
