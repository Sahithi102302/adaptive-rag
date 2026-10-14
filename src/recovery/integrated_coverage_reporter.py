
"""Phase 11.4F-6: integrated diagnostic evidence coverage.

Combines relationship coverage, joined paths, and explicit
contradiction detection without claiming semantic verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from src.recovery.contradiction_aware_validator import (
    ContradictionAwarePathValidator,
    FactContradiction,
    PathValidationDecision,
)
from src.recovery.question_coverage_evaluator import (
    CoverageDecision,
    QuestionCoverageEvaluator,
    QuestionCoverageReport,
)
from src.recovery.question_requirements import (
    QuestionRequirements,
)
from src.recovery.relational_evidence_joiner import (
    EvidencePath,
    RelationalEvidenceJoiner,
    JoinDecision,
)


class IntegratedDecision(str, Enum):
    UNIQUE_CANDIDATE = "unique_candidate"
    AMBIGUOUS = "ambiguous"
    CONFLICTING = "conflicting"
    INCOMPLETE = "incomplete"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EvidencePathSummary:
    answer_value: str | None
    bindings: tuple[tuple[str, str], ...]
    supporting_chunk_ids: tuple[str, ...]
    contradictions: tuple[FactContradiction, ...]


@dataclass(frozen=True)
class IntegratedCoverageReport:
    decision: IntegratedDecision
    independent_coverage: QuestionCoverageReport
    path_summaries: tuple[EvidencePathSummary, ...]
    candidate_answers: tuple[str, ...]
    clean_candidate_answers: tuple[str, ...]
    contradictions: tuple[FactContradiction, ...]
    explanation: str

    @property
    def eligible_for_automatic_acceptance(self) -> bool:
        # Deliberately disabled until semantic and completeness
        # verification are independently evaluated.
        return False


class IntegratedCoverageReporter:
    name = "integrated_coverage_reporter_v1"

    def __init__(self) -> None:
        self.joiner = RelationalEvidenceJoiner()
        self.validator = ContradictionAwarePathValidator()
        self.coverage_evaluator = QuestionCoverageEvaluator()

    def assess(
        self,
        requirements: QuestionRequirements,
        evidence_by_id: Mapping[str, str],
    ) -> IntegratedCoverageReport:
        if not isinstance(requirements, QuestionRequirements):
            raise TypeError("Expected QuestionRequirements.")

        joined = self.joiner.join(requirements, evidence_by_id)
        validated = self.validator.validate(
            requirements, evidence_by_id
        )

        # The independent matcher is retained as a separate
        # diagnostic. Its labels can disagree with joined paths.
        coverage_assessment = self.joiner.matcher.assess(
            requirements, evidence_by_id
        )
        independent = self.coverage_evaluator.evaluate(
            coverage_assessment
        )

        summaries = tuple(
            EvidencePathSummary(
                answer_value=item.path.binding(
                    requirements.answer_variable
                ),
                bindings=item.path.bindings,
                supporting_chunk_ids=(
                    item.path.supporting_chunk_ids
                ),
                contradictions=item.contradictions,
            )
            for item in validated.paths
        )

        if validated.decision == PathValidationDecision.CONFLICTING:
            decision = IntegratedDecision.CONFLICTING

        elif validated.decision == PathValidationDecision.AMBIGUOUS:
            decision = IntegratedDecision.AMBIGUOUS

        elif validated.decision == PathValidationDecision.UNKNOWN:
            decision = IntegratedDecision.UNKNOWN

        elif validated.decision == PathValidationDecision.NO_COMPLETE_PATH:
            decision = IntegratedDecision.INCOMPLETE

        elif (
            validated.decision
            == PathValidationDecision.UNIQUE_UNCONTRADICTED
        ):
            decision = IntegratedDecision.UNIQUE_CANDIDATE

        else:
            decision = IntegratedDecision.UNKNOWN

        # Do not use independent coverage's CONFLICTING label
        # as proof of a factual contradiction: its older
        # binding-consistency check can classify disconnected
        # entities as conflicting.
        explanation = {
            IntegratedDecision.UNIQUE_CANDIDATE: (
                "One candidate answer has a complete extracted "
                "positive path without detected explicit negation. "
                "Semantic correctness remains unverified."
            ),
            IntegratedDecision.AMBIGUOUS: (
                "Multiple candidate answers have complete "
                "uncontradicted extracted paths."
            ),
            IntegratedDecision.CONFLICTING: (
                "At least one complete extracted path contains "
                "an explicit positive/negative fact conflict."
            ),
            IntegratedDecision.INCOMPLETE: (
                "No complete extracted positive evidence path "
                "satisfies all declared relationships."
            ),
            IntegratedDecision.UNKNOWN: (
                "The restricted matcher or path validator "
                "could not establish a determinate result."
            ),
        }[decision]

        return IntegratedCoverageReport(
            decision=decision,
            independent_coverage=independent,
            path_summaries=summaries,
            candidate_answers=joined.answer_values,
            clean_candidate_answers=(
                validated.clean_answer_values
            ),
            contradictions=validated.contradictions,
            explanation=explanation,
        )
