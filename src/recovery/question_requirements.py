
"""Phase 11.4F-1: question requirements and evidence coverage contracts.

Structural validation only. Does not extract requirements, verify
semantic entailment, or authorize automatic answer acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CoverageStatus(str, Enum):
    SUPPORTED = "supported"
    MISSING = "missing"
    CONFLICTING = "conflicting"
    UNKNOWN = "unknown"


def _nonempty(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string.")


def _unique_strings(values: tuple[str, ...], field: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{field} must be a tuple.")

    for value in values:
        _nonempty(value, field)

    if len(values) != len(set(values)):
        raise ValueError(f"{field} contains duplicates.")


@dataclass(frozen=True)
class RequiredRelationship:
    relationship_id: str
    subject: str
    predicate: str
    object: str

    def __post_init__(self) -> None:
        for field in (
            "relationship_id",
            "subject",
            "predicate",
            "object",
        ):
            _nonempty(getattr(self, field), field)


@dataclass(frozen=True)
class QuestionRequirements:
    question: str
    relationships: tuple[RequiredRelationship, ...]
    answer_variable: str

    def __post_init__(self) -> None:
        _nonempty(self.question, "question")
        _nonempty(self.answer_variable, "answer_variable")

        if not isinstance(self.relationships, tuple):
            raise TypeError("relationships must be a tuple.")

        if not self.relationships:
            raise ValueError("At least one relationship is required.")

        if not all(
            isinstance(item, RequiredRelationship)
            for item in self.relationships
        ):
            raise TypeError("Invalid relationship object.")

        ids = tuple(
            item.relationship_id for item in self.relationships
        )
        _unique_strings(ids, "relationship IDs")

        variables = {
            term
            for relationship in self.relationships
            for term in (relationship.subject, relationship.object)
            if term.startswith("?")
        }

        if self.answer_variable not in variables:
            raise ValueError(
                "Answer variable must appear in a relationship."
            )

        # A variable occurring in multiple relationships represents
        # a shared entity. We do not impose a fixed hop count.


@dataclass(frozen=True)
class RelationshipCoverage:
    relationship_id: str
    status: CoverageStatus
    supporting_chunk_ids: tuple[str, ...]
    bindings: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _nonempty(self.relationship_id, "relationship_id")

        if not isinstance(self.status, CoverageStatus):
            raise TypeError("status must be CoverageStatus.")

        _unique_strings(
            self.supporting_chunk_ids,
            "supporting_chunk_ids",
        )

        if not isinstance(self.bindings, tuple):
            raise TypeError("bindings must be a tuple.")

        names = []

        for binding in self.bindings:
            if (
                not isinstance(binding, tuple)
                or len(binding) != 2
            ):
                raise TypeError(
                    "Each binding must be a (variable, value) tuple."
                )

            variable, value = binding

            _nonempty(variable, "binding variable")
            _nonempty(value, "binding value")

            if not variable.startswith("?"):
                raise ValueError(
                    "Binding names must begin with '?'."
                )

            names.append(variable)

        if len(names) != len(set(names)):
            raise ValueError("Duplicate variable bindings.")

        if self.status == CoverageStatus.SUPPORTED:
            if not self.supporting_chunk_ids:
                raise ValueError(
                    "Supported relationships require citations."
                )

        elif self.status in (
            CoverageStatus.MISSING,
            CoverageStatus.UNKNOWN,
        ):
            if self.supporting_chunk_ids or self.bindings:
                raise ValueError(
                    "Missing or unknown relationships cannot "
                    "claim evidence or entity bindings."
                )

        elif self.status == CoverageStatus.CONFLICTING:
            if len(self.supporting_chunk_ids) < 2:
                raise ValueError(
                    "Conflicting relationships require "
                    "at least two cited chunks."
                )


@dataclass(frozen=True)
class QuestionCoverageAssessment:
    requirements: QuestionRequirements
    coverage: tuple[RelationshipCoverage, ...]
    assessed_chunk_ids: tuple[str, ...]
    explanation: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.requirements, QuestionRequirements
        ):
            raise TypeError("Invalid question requirements.")

        if not isinstance(self.coverage, tuple):
            raise TypeError("coverage must be a tuple.")

        if not all(
            isinstance(item, RelationshipCoverage)
            for item in self.coverage
        ):
            raise TypeError("Invalid coverage object.")

        _unique_strings(
            self.assessed_chunk_ids,
            "assessed_chunk_ids",
        )
        _nonempty(self.explanation, "explanation")

        required_ids = {
            item.relationship_id
            for item in self.requirements.relationships
        }

        covered_ids = [
            item.relationship_id for item in self.coverage
        ]

        if (
            len(covered_ids) != len(set(covered_ids))
            or set(covered_ids) != required_ids
        ):
            raise ValueError(
                "Coverage must contain exactly one result "
                "for each required relationship."
            )

        allowed_ids = set(self.assessed_chunk_ids)

        for item in self.coverage:
            if not set(
                item.supporting_chunk_ids
            ).issubset(allowed_ids):
                raise ValueError(
                    "Coverage cites unknown evidence."
                )

            requirement = next(
                relationship
                for relationship in self.requirements.relationships
                if relationship.relationship_id
                == item.relationship_id
            )

            permitted_variables = {
                term
                for term in (
                    requirement.subject,
                    requirement.object,
                )
                if term.startswith("?")
            }

            for variable, _ in item.bindings:
                if variable not in permitted_variables:
                    raise ValueError(
                        "Binding references a variable "
                        "not present in its relationship."
                    )

    @property
    def has_missing_links(self) -> bool:
        return any(
            item.status == CoverageStatus.MISSING
            for item in self.coverage
        )

    @property
    def has_conflicts(self) -> bool:
        return any(
            item.status == CoverageStatus.CONFLICTING
            for item in self.coverage
        )

    @property
    def all_relationships_supported(self) -> bool:
        return all(
            item.status == CoverageStatus.SUPPORTED
            for item in self.coverage
        )

    @property
    def bindings_consistent(self) -> bool:
        seen: dict[str, str] = {}

        for item in self.coverage:
            if item.status != CoverageStatus.SUPPORTED:
                continue

            for variable, value in item.bindings:
                if variable in seen and seen[variable] != value:
                    return False

                seen[variable] = value

        return True

    @property
    def structurally_complete(self) -> bool:
        """Necessary structural conditions, not semantic proof."""
        if not self.all_relationships_supported:
            return False

        if not self.bindings_consistent:
            return False

        required_variables = {
            term
            for relationship in self.requirements.relationships
            for term in (
                relationship.subject,
                relationship.object,
            )
            if term.startswith("?")
        }

        bound_variables = {
            variable
            for item in self.coverage
            for variable, _ in item.bindings
        }

        return required_variables.issubset(bound_variables)
