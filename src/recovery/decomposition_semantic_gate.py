
"""Phase 11.4G-10: controlled paraphrase generalization.

A conservative, task-specific consistency gate for questions
about the educational institution attended by a book's author.

This is not a general semantic entailment verifier.
It checks question grammar, entity identity, graph structure,
relationship direction, and answer-variable consistency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from src.recovery.question_requirements import (
    QuestionRequirements,
)


class SemanticGateStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True)
class SemanticGateResult:
    status: SemanticGateStatus
    reason: str

    @property
    def accepted(self) -> bool:
        return self.status == SemanticGateStatus.ACCEPTED


class DecompositionSemanticGate:
    name = "decomposition_semantic_gate_v2"

    _QUESTION_PATTERNS = (
        # Original supported grammar.
        re.compile(
            r"^(?:which|what) (?:university|college) "
            r"did the (?:author|writer) of (?P<title>.+?) "
            r"attend\??$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^where did the (?:author|writer) of "
            r"(?P<title>.+?) study\??$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^at which (?:university|college) did the "
            r"(?:author|writer) of (?P<title>.+?) study\??$",
            re.IGNORECASE,
        ),

        # Passive voice.
        re.compile(
            r"^which (?:university|college) was attended by "
            r"the (?:author|writer) of (?P<title>.+?)\?$",
            re.IGNORECASE,
        ),

        # Relative-clause authorship and "go to".
        re.compile(
            r"^(?:which|what) (?:university|college) did "
            r"the person who wrote (?P<title>.+?) "
            r"go to\?$",
            re.IGNORECASE,
        ),

        # Imperative answer request.
        re.compile(
            r"^name the (?:university|college) attended by "
            r"the (?:author|writer) of (?P<title>.+?)\.$",
            re.IGNORECASE,
        ),

        # Answer-first / inverted formulation.
        re.compile(
            r"^the (?:author|writer) of (?P<title>.+?) "
            r"studied at which (?:university|college)\?$",
            re.IGNORECASE,
        ),
    )

    @staticmethod
    def _reject(reason: str) -> SemanticGateResult:
        return SemanticGateResult(
            status=SemanticGateStatus.REJECTED,
            reason=reason,
        )

    @staticmethod
    def _normalize_entity(value: str) -> str:
        value = " ".join(value.split()).strip()

        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in ("'", '"')
        ):
            value = value[1:-1].strip()

        return value.casefold()

    def check(
        self,
        question: str,
        requirements: QuestionRequirements,
    ) -> SemanticGateResult:
        if not isinstance(question, str) or not question.strip():
            return self._reject("Invalid question.")

        if not isinstance(requirements, QuestionRequirements):
            return self._reject("Invalid requirements object.")

        normalized_question = " ".join(question.split())

        match = None

        for pattern in self._QUESTION_PATTERNS:
            match = pattern.fullmatch(normalized_question)

            if match is not None:
                break

        if match is None:
            return self._reject(
                "Question is outside the supported two-hop "
                "author/education grammar."
            )

        expected_title = self._normalize_entity(
            match.group("title")
        )

        if not expected_title:
            return self._reject("Question has no book title.")

        relations = requirements.relationships

        if len(relations) != 2:
            return self._reject(
                "Expected exactly two relationships."
            )

        authored = [
            relation
            for relation in relations
            if relation.predicate == "authored_by"
        ]

        attended = [
            relation
            for relation in relations
            if relation.predicate == "attended"
        ]

        if len(authored) != 1 or len(attended) != 1:
            return self._reject(
                "Proposal has incorrect predicates."
            )

        first = authored[0]
        second = attended[0]

        if self._normalize_entity(first.subject) != expected_title:
            return self._reject(
                "Proposal book entity differs from question."
            )

        author_variable = first.object

        if not author_variable.startswith("?"):
            return self._reject(
                "Authorship must bind an author variable."
            )

        if second.subject != author_variable:
            return self._reject(
                "Education relation does not use "
                "the bound author variable."
            )

        if not second.object.startswith("?"):
            return self._reject(
                "Education relation must bind an answer variable."
            )

        if second.object == author_variable:
            return self._reject(
                "Author and answer variables must be distinct."
            )

        if requirements.answer_variable != second.object:
            return self._reject(
                "Answer target differs from education object."
            )

        return SemanticGateResult(
            status=SemanticGateStatus.ACCEPTED,
            reason=(
                "Proposal matches the supported question "
                "grammar and expected two-hop structure."
            ),
        )
