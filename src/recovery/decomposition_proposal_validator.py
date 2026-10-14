
"""Phase 11.4G-4: validate LLM-proposed question decompositions.

Structural validation only. A valid proposal is not necessarily
a faithful interpretation of the original question.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping
import re

from src.recovery.question_requirements import (
    QuestionRequirements,
    RequiredRelationship,
)


class ProposalStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"


@dataclass(frozen=True)
class ProposalValidation:
    status: ProposalStatus
    requirements: QuestionRequirements | None
    explanation: str

    @property
    def is_valid(self) -> bool:
        return (
            self.status == ProposalStatus.VALID
            and self.requirements is not None
        )


class DecompositionProposalValidator:
    name = "decomposition_proposal_validator_v1"

    SUPPORTED_PREDICATES = frozenset({
        "authored_by",
        "attended",
    })

    MAX_RELATIONSHIPS = 4
    MAX_FIELD_LENGTH = 256

    _VARIABLE = re.compile(r"^\?[a-z][a-z0-9_]*$")

    @staticmethod
    def _invalid(reason: str) -> ProposalValidation:
        return ProposalValidation(
            status=ProposalStatus.INVALID,
            requirements=None,
            explanation=reason,
        )

    def validate(
        self,
        question: str,
        proposal: Any,
    ) -> ProposalValidation:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a nonempty string.")

        if not isinstance(proposal, Mapping):
            return self._invalid("Proposal must be a JSON object.")

        if set(proposal) != {"relationships", "answer_variable"}:
            return self._invalid(
                "Proposal must contain exactly relationships "
                "and answer_variable."
            )

        raw_relations = proposal["relationships"]
        answer = proposal["answer_variable"]

        if (
            not isinstance(raw_relations, list)
            or not 1 <= len(raw_relations) <= self.MAX_RELATIONSHIPS
        ):
            return self._invalid(
                "Expected between 1 and 4 relationships."
            )

        if (
            not isinstance(answer, str)
            or not self._VARIABLE.fullmatch(answer)
        ):
            return self._invalid("Invalid answer variable.")

        relations = []

        for index, raw in enumerate(raw_relations, start=1):
            if not isinstance(raw, Mapping):
                return self._invalid(
                    f"Relationship {index} is not an object."
                )

            if set(raw) != {"subject", "predicate", "object"}:
                return self._invalid(
                    f"Relationship {index} has invalid fields."
                )

            subject = raw["subject"]
            predicate = raw["predicate"]
            obj = raw["object"]

            if not all(
                isinstance(value, str)
                and value.strip()
                and len(value) <= self.MAX_FIELD_LENGTH
                for value in (subject, predicate, obj)
            ):
                return self._invalid(
                    f"Relationship {index} has invalid values."
                )

            subject = subject.strip()
            predicate = predicate.strip()
            obj = obj.strip()

            if predicate not in self.SUPPORTED_PREDICATES:
                return self._invalid(
                    f"Unsupported predicate: {predicate}"
                )

            for term in (subject, obj):
                if term.startswith("?") and not self._VARIABLE.fullmatch(term):
                    return self._invalid(
                        f"Invalid variable name: {term}"
                    )

            relations.append(
                RequiredRelationship(
                    relationship_id=f"r{index}",
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                )
            )

        # Require all relationships to form one connected graph.
        # Edges connect shared exact entity names or variables.
        seen = {0}
        pending = [0]

        while pending:
            current = pending.pop()
            current_terms = {
                relations[current].subject,
                relations[current].object,
            }

            for index, relation in enumerate(relations):
                if index in seen:
                    continue

                other_terms = {relation.subject, relation.object}

                if current_terms & other_terms:
                    seen.add(index)
                    pending.append(index)

        if len(seen) != len(relations):
            return self._invalid(
                "Relationships do not form one connected graph."
            )

        try:
            requirements = QuestionRequirements(
                question=question,
                relationships=tuple(relations),
                answer_variable=answer,
            )
        except (ValueError, TypeError) as exc:
            return self._invalid(str(exc))

        return ProposalValidation(
            status=ProposalStatus.VALID,
            requirements=requirements,
            explanation=(
                "Proposal passed structural validation. "
                "Semantic fidelity remains unverified."
            ),
        )
