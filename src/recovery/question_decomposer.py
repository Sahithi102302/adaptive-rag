
"""Phase 11.4G-1: conservative question decomposition baseline.

Recognizes a restricted two-hop author/university question.
Does not use retrieved evidence, gold supporting facts, or an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

from src.recovery.question_requirements import (
    QuestionRequirements,
    RequiredRelationship,
)


class DecompositionStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class QuestionDecomposition:
    status: DecompositionStatus
    requirements: QuestionRequirements | None
    explanation: str

    @property
    def is_usable(self) -> bool:
        return (
            self.status == DecompositionStatus.SUPPORTED
            and self.requirements is not None
        )


class ConservativeQuestionDecomposer:
    name = "conservative_question_decomposer_v1"

    _AUTHOR_UNIVERSITY = re.compile(
        r"^Which university did the author of "
        r"(?P<title>.+?) attend\?$",
        flags=re.IGNORECASE,
    )

    def decompose(
        self,
        question: str,
    ) -> QuestionDecomposition:
        if not isinstance(question, str):
            raise TypeError("Question must be a string.")

        question = " ".join(question.split())

        if not question:
            raise ValueError("Question cannot be empty.")

        match = self._AUTHOR_UNIVERSITY.fullmatch(question)

        if match is None:
            return QuestionDecomposition(
                status=DecompositionStatus.UNSUPPORTED,
                requirements=None,
                explanation=(
                    "Question does not match the supported "
                    "two-hop author/university template."
                ),
            )

        title = match.group("title").strip()

        # Quotation marks are optional around a book title.
        if (
            len(title) >= 2
            and title[0] == title[-1]
            and title[0] in ("'", '"')
        ):
            title = title[1:-1].strip()

        if not title:
            return QuestionDecomposition(
                status=DecompositionStatus.UNSUPPORTED,
                requirements=None,
                explanation="No valid book title was extracted.",
            )

        requirements = QuestionRequirements(
            question=question,
            relationships=(
                RequiredRelationship(
                    relationship_id="r1",
                    subject=title,
                    predicate="authored_by",
                    object="?author",
                ),
                RequiredRelationship(
                    relationship_id="r2",
                    subject="?author",
                    predicate="attended",
                    object="?university",
                ),
            ),
            answer_variable="?university",
        )

        return QuestionDecomposition(
            status=DecompositionStatus.SUPPORTED,
            requirements=requirements,
            explanation=(
                "Matched the supported two-hop "
                "author/university question template."
            ),
        )
