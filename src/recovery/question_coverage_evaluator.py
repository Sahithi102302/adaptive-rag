
"""Phase 11.4F-2: deterministic question coverage evaluation.

Summarizes structural evidence coverage, missing links,
conflicts, unknown relationships, and inconsistent bindings.

Does not independently verify semantic entailment,
question decomposition, or factual correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.recovery.question_requirements import (
    CoverageStatus,
    QuestionCoverageAssessment,
)


class CoverageDecision(str, Enum):
    STRUCTURALLY_COMPLETE = "structurally_complete"
    INCOMPLETE = "incomplete"
    CONFLICTING = "conflicting"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class QuestionCoverageReport:
    decision: CoverageDecision
    supported_relationship_ids: tuple[str, ...]
    missing_relationship_ids: tuple[str, ...]
    conflicting_relationship_ids: tuple[str, ...]
    unknown_relationship_ids: tuple[str, ...]
    inconsistent_variables: tuple[str, ...]
    unbound_variables: tuple[str, ...]
    answer_value: str | None
    explanation: str

    @property
    def is_structurally_complete(self) -> bool:
        return (
            self.decision
            == CoverageDecision.STRUCTURALLY_COMPLETE
        )


class QuestionCoverageEvaluator:
    name = "question_coverage_evaluator_v1"

    def evaluate(
        self,
        assessment: QuestionCoverageAssessment,
    ) -> QuestionCoverageReport:
        if not isinstance(
            assessment, QuestionCoverageAssessment
        ):
            raise TypeError(
                "Expected QuestionCoverageAssessment."
            )

        groups = {
            status: []
            for status in CoverageStatus
        }

        values_by_variable: dict[str, set[str]] = {}

        for item in assessment.coverage:
            groups[item.status].append(item.relationship_id)

            if item.status == CoverageStatus.SUPPORTED:
                for variable, value in item.bindings:
                    values_by_variable.setdefault(
                        variable, set()
                    ).add(value)

        inconsistent = tuple(
            sorted(
                variable
                for variable, values
                in values_by_variable.items()
                if len(values) > 1
            )
        )

        required_variables = {
            term
            for relation in assessment.requirements.relationships
            for term in (relation.subject, relation.object)
            if term.startswith("?")
        }

        unbound = tuple(
            sorted(
                required_variables
                - set(values_by_variable)
            )
        )

        answer_variable = (
            assessment.requirements.answer_variable
        )

        answer_values = values_by_variable.get(
            answer_variable, set()
        )

        answer_value = (
            next(iter(answer_values))
            if len(answer_values) == 1
            else None
        )

        if groups[CoverageStatus.CONFLICTING] or inconsistent:
            decision = CoverageDecision.CONFLICTING

        elif groups[CoverageStatus.MISSING]:
            decision = CoverageDecision.INCOMPLETE

        elif groups[CoverageStatus.UNKNOWN] or unbound:
            decision = CoverageDecision.UNKNOWN

        else:
            decision = CoverageDecision.STRUCTURALLY_COMPLETE

        explanation = {
            CoverageDecision.STRUCTURALLY_COMPLETE: (
                "All declared relationships are marked supported; "
                "required variables are bound consistently. "
                "Semantic validity remains unverified."
            ),
            CoverageDecision.INCOMPLETE: (
                "At least one declared required relationship "
                "is missing."
            ),
            CoverageDecision.CONFLICTING: (
                "Conflicting relationships or inconsistent "
                "entity bindings were reported."
            ),
            CoverageDecision.UNKNOWN: (
                "Some required relationships or variable "
                "bindings remain unresolved."
            ),
        }[decision]

        return QuestionCoverageReport(
            decision=decision,
            supported_relationship_ids=tuple(
                groups[CoverageStatus.SUPPORTED]
            ),
            missing_relationship_ids=tuple(
                groups[CoverageStatus.MISSING]
            ),
            conflicting_relationship_ids=tuple(
                groups[CoverageStatus.CONFLICTING]
            ),
            unknown_relationship_ids=tuple(
                groups[CoverageStatus.UNKNOWN]
            ),
            inconsistent_variables=inconsistent,
            unbound_variables=unbound,
            answer_value=answer_value,
            explanation=explanation,
        )
