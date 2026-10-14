
"""Phase 11.4F-4: deterministic relational evidence joins.

Joins extracted positive facts using shared variable bindings.
Diagnostic only: does not establish semantic entailment, detect all
contradictions, or authorize automatic recovery acceptance.
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
from src.recovery.question_requirements import (
    QuestionRequirements,
    RequiredRelationship,
)


class JoinDecision(str, Enum):
    UNIQUE_ANSWER = "unique_answer"
    AMBIGUOUS = "ambiguous"
    NO_COMPLETE_PATH = "no_complete_path"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EvidencePath:
    bindings: tuple[tuple[str, str], ...]
    supporting_chunk_ids: tuple[str, ...]
    fact_indices: tuple[int, ...]

    def binding(self, variable: str) -> str | None:
        return dict(self.bindings).get(variable)


@dataclass(frozen=True)
class RelationalJoinReport:
    decision: JoinDecision
    paths: tuple[EvidencePath, ...]
    answer_values: tuple[str, ...]
    unmatched_relationship_ids: tuple[str, ...]
    explanation: str


class RelationalEvidenceJoiner:
    name = "relational_evidence_joiner_v1"

    def __init__(self) -> None:
        self.matcher = EvidenceRelationshipMatcher()

    @staticmethod
    def _candidate_bindings(
        relation: RequiredRelationship,
        fact: ExtractedFact,
    ) -> dict[str, str] | None:
        if fact.negated or relation.predicate != fact.predicate:
            return None

        bindings: dict[str, str] = {}

        for required, observed in (
            (relation.subject, fact.subject),
            (relation.object, fact.object),
        ):
            if required.startswith("?"):
                if (
                    required in bindings
                    and _normalize(bindings[required])
                    != _normalize(observed)
                ):
                    return None
                bindings[required] = observed
            elif _normalize(required) != _normalize(observed):
                return None

        return bindings

    @staticmethod
    def _merge(
        existing: Mapping[str, str],
        candidate: Mapping[str, str],
    ) -> dict[str, str] | None:
        merged = dict(existing)

        for variable, value in candidate.items():
            if (
                variable in merged
                and _normalize(merged[variable])
                != _normalize(value)
            ):
                return None
            merged.setdefault(variable, value)

        return merged

    def join(
        self,
        requirements: QuestionRequirements,
        evidence_by_id: Mapping[str, str],
    ) -> RelationalJoinReport:
        if not isinstance(requirements, QuestionRequirements):
            raise TypeError("Expected QuestionRequirements.")

        facts = self.matcher.extract(evidence_by_id)

        unsupported = tuple(
            relation.relationship_id
            for relation in requirements.relationships
            if relation.predicate
            not in self.matcher._SUPPORTED_PREDICATES
        )

        if unsupported:
            return RelationalJoinReport(
                decision=JoinDecision.UNKNOWN,
                paths=(),
                answer_values=(),
                unmatched_relationship_ids=unsupported,
                explanation=(
                    "One or more predicates are outside "
                    "the supported extraction grammar."
                ),
            )

        # State: (variable bindings, cited chunks, fact indices).
        states: list[
            tuple[dict[str, str], tuple[str, ...], tuple[int, ...]]
        ] = [({}, (), ())]

        unmatched: list[str] = []

        for relation in requirements.relationships:
            candidates = [
                (index, fact, bindings)
                for index, fact in enumerate(facts)
                if (
                    bindings := self._candidate_bindings(
                        relation, fact
                    )
                ) is not None
            ]

            if not candidates:
                unmatched.append(relation.relationship_id)

            next_states = []

            for bindings, citations, indices in states:
                for index, fact, candidate in candidates:
                    merged = self._merge(bindings, candidate)
                    if merged is None:
                        continue

                    next_states.append(
                        (
                            merged,
                            tuple(dict.fromkeys(
                                (*citations, fact.chunk_id)
                            )),
                            (*indices, index),
                        )
                    )

            states = next_states

        if not states:
            return RelationalJoinReport(
                decision=JoinDecision.NO_COMPLETE_PATH,
                paths=(),
                answer_values=(),
                unmatched_relationship_ids=tuple(unmatched),
                explanation=(
                    "No complete positive evidence path satisfies "
                    "all declared relationships and shared bindings. "
                    "This does not establish that the answer is false."
                ),
            )

        paths = tuple(
            EvidencePath(
                bindings=tuple(sorted(bindings.items())),
                supporting_chunk_ids=citations,
                fact_indices=indices,
            )
            for bindings, citations, indices in states
        )

        answer_variable = requirements.answer_variable

        # Compare answer values by normalized identity while
        # preserving the original evidence wording for display.
        answer_map: dict[str, str] = {}

        for path in paths:
            answer = path.binding(answer_variable)
            if answer is not None:
                answer_map.setdefault(_normalize(answer), answer)

        answers = tuple(answer_map.values())

        if not answers:
            decision = JoinDecision.UNKNOWN
            explanation = "No complete path binds the answer variable."
        elif len(answers) == 1:
            decision = JoinDecision.UNIQUE_ANSWER
            explanation = (
                "Complete positive evidence paths agree on one "
                "answer. Contradiction coverage and semantic "
                "validity have not been established."
            )
        else:
            decision = JoinDecision.AMBIGUOUS
            explanation = (
                "Multiple distinct answer values have complete "
                "positive evidence paths."
            )

        return RelationalJoinReport(
            decision=decision,
            paths=paths,
            answer_values=answers,
            unmatched_relationship_ids=tuple(unmatched),
            explanation=explanation,
        )
