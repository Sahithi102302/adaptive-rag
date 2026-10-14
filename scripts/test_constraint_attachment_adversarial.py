
"""Phase 11.4G-12H-11H: adversarial attachment contract tests.

Demonstrates that structurally valid constraint attachments
may still be semantically incorrect or incomplete.

These are representation tests, not semantic verification.
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


def make_requirements() -> QuestionRequirements:
    return QuestionRequirements(
        question=(
            "Which position did the actress who portrayed "
            "Corliss Archer in Kiss and Tell hold before 1998?"
        ),
        relationships=(
            RequiredRelationship(
                relationship_id="r1",
                subject="Corliss Archer",
                predicate="portrayed_by",
                object="?actress",
            ),
            RequiredRelationship(
                relationship_id="r2",
                subject="?actress",
                predicate="held_position",
                object="?position",
            ),
        ),
        answer_variable="?position",
    )


def constraint(
    constraint_id: str,
    kind: AttachmentConstraintKind,
    operator: str,
    value: str,
    relationship_id: str,
) -> AttachedConstraint:
    return AttachedConstraint(
        constraint_id=constraint_id,
        kind=kind,
        operator=operator,
        value=value,
        relationship_id=relationship_id,
    )


def main() -> None:
    requirements = make_requirements()

    cases = (
        (
            "H01_reversed_temporal_operator",
            (
                constraint(
                    "c1",
                    AttachmentConstraintKind.TEMPORAL,
                    "after",
                    "1998",
                    "r2",
                ),
            ),
        ),
        (
            "H02_wrong_existing_relationship",
            (
                constraint(
                    "c1",
                    AttachmentConstraintKind.TEMPORAL,
                    "before",
                    "1998",
                    "r1",
                ),
            ),
        ),
        (
            "H03_missing_constraint",
            (),
        ),
        (
            "H04_incorrect_contextual_entity",
            (
                constraint(
                    "c1",
                    AttachmentConstraintKind.CONTEXTUAL,
                    "in_film",
                    "Another Film",
                    "r1",
                ),
            ),
        ),
    )

    for case_id, constraints in cases:
        result = ConstraintAttachmentSet(
            requirements=requirements,
            constraints=constraints,
        )

        assert result.constraints == constraints

        print(
            f"PASS {case_id}: structurally valid; "
            "semantic fidelity not established"
        )

    print(
        f"\nAdversarial attachment tests: "
        f"{len(cases)}/{len(cases)}"
    )


if __name__ == "__main__":
    main()
