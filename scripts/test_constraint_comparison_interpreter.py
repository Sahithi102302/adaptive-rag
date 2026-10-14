
"""Tests for Phase 11.4G-12H-11K comparison interpretation."""

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
from src.recovery.constraint_comparison_interpreter import (
    ComparisonInterpretationStatus,
    ConstraintComparisonInterpreter,
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
    operator: str = "before",
) -> AttachedConstraint:
    return AttachedConstraint(
        constraint_id=constraint_id,
        kind=AttachmentConstraintKind.TEMPORAL,
        operator=operator,
        value="1998",
        relationship_id="r1",
    )


def main() -> None:
    requirements = make_requirements()
    interpreter = ConstraintComparisonInterpreter()

    cases = (
        (
            "K01_exact_nonempty_agreement",
            (make_constraint(),),
            (make_constraint(),),
            ComparisonInterpretationStatus.AGREEMENT_OBSERVED,
        ),
        (
            "K02_reversed_operator",
            (make_constraint(),),
            (make_constraint(operator="after"),),
            ComparisonInterpretationStatus.DISAGREEMENT_OBSERVED,
        ),
        (
            "K03_missing_constraint",
            (make_constraint(),),
            (),
            ComparisonInterpretationStatus.DISAGREEMENT_OBSERVED,
        ),
        (
            "K04_both_empty",
            (),
            (),
            ComparisonInterpretationStatus.INSUFFICIENT_REFERENCE,
        ),
        (
            "K05_empty_reference_extra_proposal",
            (),
            (make_constraint(),),
            ComparisonInterpretationStatus.INSUFFICIENT_REFERENCE,
        ),
        (
            "K06_different_constraint_id",
            (make_constraint(),),
            (make_constraint(constraint_id="c99"),),
            ComparisonInterpretationStatus.DISAGREEMENT_OBSERVED,
        ),
    )

    for case_id, reference_constraints, proposed_constraints, expected in cases:
        reference = ConstraintAttachmentSet(
            requirements=requirements,
            constraints=reference_constraints,
        )
        proposed = ConstraintAttachmentSet(
            requirements=requirements,
            constraints=proposed_constraints,
        )

        result = interpreter.interpret(reference, proposed)

        assert result.status == expected, (
            f"{case_id}: expected {expected}, got {result.status}"
        )

        print(
            f"PASS {case_id}: {result.status.value}"
        )

    print(
        f"\nComparison interpretation tests: "
        f"{len(cases)}/{len(cases)}"
    )


if __name__ == "__main__":
    main()
