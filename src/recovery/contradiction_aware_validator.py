
"""Phase 11.4F-5: contradiction-aware evidence-path validation.

Detects explicit negations of positive facts used in joined paths.
Diagnostic only; not a general natural-language contradiction detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from src.recovery.evidence_relationship_matcher import (
    EvidenceRelationshipMatcher,
    ExtractedFact,
    _normalize,
)
from src.recovery.question_requirements import QuestionRequirements
from src.recovery.relational_evidence_joiner import (
    EvidencePath,
    JoinDecision,
    RelationalEvidenceJoiner,
)


class PathValidationDecision(str, Enum):
    UNIQUE_UNCONTRADICTED = "unique_uncontradicted"
    AMBIGUOUS = "ambiguous"
    CONFLICTING = "conflicting"
    NO_COMPLETE_PATH = "no_complete_path"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FactContradiction:
    positive_fact_index: int
    negative_fact_index: int
    positive_chunk_id: str
    negative_chunk_id: str
    predicate: str
    subject: str
    object: str


@dataclass(frozen=True)
class ValidatedEvidencePath:
    path: EvidencePath
    contradictions: tuple[FactContradiction, ...]

    @property
    def is_uncontradicted(self) -> bool:
        return not self.contradictions


@dataclass(frozen=True)
class PathValidationReport:
    decision: PathValidationDecision
    paths: tuple[ValidatedEvidencePath, ...]
    clean_answer_values: tuple[str, ...]
    contradictions: tuple[FactContradiction, ...]
    explanation: str


class ContradictionAwarePathValidator:
    name = "contradiction_aware_path_validator_v1"

    def __init__(self) -> None:
        self.matcher = EvidenceRelationshipMatcher()
        self.joiner = RelationalEvidenceJoiner()

    @staticmethod
    def _opposes(
        positive: ExtractedFact,
        negative: ExtractedFact,
    ) -> bool:
        return (
            not positive.negated
            and negative.negated
            and positive.predicate == negative.predicate
            and _normalize(positive.subject)
            == _normalize(negative.subject)
            and _normalize(positive.object)
            == _normalize(negative.object)
        )

    def validate(
        self,
        requirements: QuestionRequirements,
        evidence_by_id: Mapping[str, str],
    ) -> PathValidationReport:
        # Join first; this also validates the input types.
        joined = self.joiner.join(
            requirements,
            evidence_by_id,
        )

        if joined.decision == JoinDecision.UNKNOWN:
            return PathValidationReport(
                decision=PathValidationDecision.UNKNOWN,
                paths=(),
                clean_answer_values=(),
                contradictions=(),
                explanation=(
                    "The positive-path join could not resolve "
                    "all required relationships."
                ),
            )

        if joined.decision == JoinDecision.NO_COMPLETE_PATH:
            return PathValidationReport(
                decision=PathValidationDecision.NO_COMPLETE_PATH,
                paths=(),
                clean_answer_values=(),
                contradictions=(),
                explanation=(
                    "No complete positive path was extracted. "
                    "Contradictions outside complete paths "
                    "have not been evaluated."
                ),
            )

        # The joiner and validator use the same deterministic
        # extraction order; fact indices refer to this sequence.
        facts = self.matcher.extract(evidence_by_id)

        negative_facts = [
            (index, fact)
            for index, fact in enumerate(facts)
            if fact.negated
        ]

        validated_paths = []
        all_contradictions = {}

        for path in joined.paths:
            path_conflicts = {}

            for positive_index in path.fact_indices:
                positive = facts[positive_index]

                for negative_index, negative in negative_facts:
                    if not self._opposes(positive, negative):
                        continue

                    conflict = FactContradiction(
                        positive_fact_index=positive_index,
                        negative_fact_index=negative_index,
                        positive_chunk_id=positive.chunk_id,
                        negative_chunk_id=negative.chunk_id,
                        predicate=positive.predicate,
                        subject=positive.subject,
                        object=positive.object,
                    )

                    key = (positive_index, negative_index)
                    path_conflicts[key] = conflict
                    all_contradictions[key] = conflict

            validated_paths.append(
                ValidatedEvidencePath(
                    path=path,
                    contradictions=tuple(path_conflicts.values()),
                )
            )

        clean_paths = [
            item
            for item in validated_paths
            if item.is_uncontradicted
        ]

        answers_by_normalized_value: dict[str, str] = {}

        for item in clean_paths:
            answer = item.path.binding(
                requirements.answer_variable
            )
            if answer is not None:
                answers_by_normalized_value.setdefault(
                    _normalize(answer),
                    answer,
                )

        clean_answers = tuple(
            answers_by_normalized_value.values()
        )

        if all_contradictions:
            # A detected conflict prevents a clean overall
            # decision even if another path remains available.
            decision = PathValidationDecision.CONFLICTING
            explanation = (
                "At least one complete evidence path contains "
                "an explicit positive/negative fact conflict. "
                "Clean alternatives, if any, are reported "
                "separately."
            )
        elif len(clean_answers) > 1:
            decision = PathValidationDecision.AMBIGUOUS
            explanation = (
                "Multiple distinct answers have complete paths "
                "without detected explicit contradictions."
            )
        elif len(clean_answers) == 1:
            decision = (
                PathValidationDecision.UNIQUE_UNCONTRADICTED
            )
            explanation = (
                "One answer has complete positive evidence paths "
                "and no matching explicit negation was extracted. "
                "This is not proof of semantic correctness."
            )
        else:
            decision = PathValidationDecision.UNKNOWN
            explanation = (
                "No uncontradicted path with a bound answer "
                "was available."
            )

        return PathValidationReport(
            decision=decision,
            paths=tuple(validated_paths),
            clean_answer_values=clean_answers,
            contradictions=tuple(
                all_contradictions.values()
            ),
            explanation=explanation,
        )
