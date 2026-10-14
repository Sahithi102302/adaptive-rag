
"""Phase 11.4G-12H-11G: constraint attachment contracts.

Represents explicitly supplied constraints and their attachment
to required relationships.

Structural validation only:
- Does not extract constraints from natural language.
- Does not verify that attachment is semantically correct.
- Does not establish decomposition fidelity.
- Does not authorize acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.recovery.question_requirements import (
    QuestionRequirements,
)


class AttachmentConstraintKind(str, Enum):
    NEGATION = "negation"
    TEMPORAL = "temporal"
    COMPARISON = "comparison"
    SUPERLATIVE = "superlative"
    CONTEXTUAL = "contextual"
    UNCERTAINTY = "uncertainty"


@dataclass(frozen=True)
class AttachedConstraint:
    constraint_id: str
    kind: AttachmentConstraintKind
    operator: str
    value: str
    relationship_id: str

    def __post_init__(self) -> None:
        for field in (
            "constraint_id",
            "operator",
            "value",
            "relationship_id",
        ):
            item = getattr(self, field)

            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"{field} must be a nonempty string."
                )

        if not isinstance(self.kind, AttachmentConstraintKind):
            raise TypeError(
                "kind must be AttachmentConstraintKind."
            )


@dataclass(frozen=True)
class ConstraintAttachmentSet:
    requirements: QuestionRequirements
    constraints: tuple[AttachedConstraint, ...]

    def __post_init__(self) -> None:
        if not isinstance(
            self.requirements,
            QuestionRequirements,
        ):
            raise TypeError(
                "requirements must be QuestionRequirements."
            )

        if not isinstance(self.constraints, tuple):
            raise TypeError(
                "constraints must be a tuple."
            )

        if not all(
            isinstance(item, AttachedConstraint)
            for item in self.constraints
        ):
            raise TypeError(
                "Invalid attached constraint."
            )

        relationship_ids = {
            relationship.relationship_id
            for relationship in self.requirements.relationships
        }

        constraint_ids = set()

        for constraint in self.constraints:
            if constraint.constraint_id in constraint_ids:
                raise ValueError(
                    "Duplicate constraint ID."
                )

            constraint_ids.add(constraint.constraint_id)

            if constraint.relationship_id not in relationship_ids:
                raise ValueError(
                    "Constraint references an unknown relationship."
                )

    @property
    def has_constraints(self) -> bool:
        return bool(self.constraints)
