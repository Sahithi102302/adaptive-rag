
"""Phase 11.4F-3: conservative evidence-to-relationship matcher.

Diagnostic baseline for a small set of explicit English fact patterns.
No LLM, gold supporting facts, paraphrase inference, or fuzzy matching.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass

from src.recovery.question_requirements import (
    CoverageStatus,
    QuestionCoverageAssessment,
    QuestionRequirements,
    RelationshipCoverage,
)


@dataclass(frozen=True)
class ExtractedFact:
    subject: str
    predicate: str
    object: str
    chunk_id: str
    negated: bool = False


def _normalize(text: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", text).casefold().split()
    ).strip(" .!?")


def _sentences(text: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n+", text)
        if part.strip()
    ]


class EvidenceRelationshipMatcher:
    name = "explicit_relationship_matcher_v1"

    # These are deliberately narrow extraction patterns.
    _PATTERNS = (
        (
            "authored_by",
            False,
            re.compile(
                r"^(.+?) was written by (.+?)\.?$",
                re.IGNORECASE,
            ),
        ),
        (
            "attended",
            True,
            re.compile(
                r"^(.+?) never attended (.+?)\.?$",
                re.IGNORECASE,
            ),
        ),
        (
            "attended",
            False,
            re.compile(
                r"^(.+?) attended (.+?)\.?$",
                re.IGNORECASE,
            ),
        ),
    )

    _SUPPORTED_PREDICATES = frozenset(
        {"authored_by", "attended"}
    )

    def extract(
        self,
        evidence_by_id: Mapping[str, str],
    ) -> tuple[ExtractedFact, ...]:
        if not isinstance(evidence_by_id, Mapping):
            raise TypeError("Evidence must be a mapping.")

        facts: list[ExtractedFact] = []

        for chunk_id, passage in evidence_by_id.items():
            if (
                not isinstance(chunk_id, str)
                or not chunk_id.strip()
                or not isinstance(passage, str)
            ):
                raise ValueError(
                    "Evidence requires nonempty string IDs "
                    "and string passages."
                )

            for sentence in _sentences(passage):
                for predicate, negated, pattern in self._PATTERNS:
                    match = pattern.fullmatch(sentence)
                    if match is None:
                        continue

                    subject, obj = (
                        value.strip(" .!?")
                        for value in match.groups()
                    )

                    if subject and obj:
                        facts.append(
                            ExtractedFact(
                                subject=subject,
                                predicate=predicate,
                                object=obj,
                                chunk_id=chunk_id,
                                negated=negated,
                            )
                        )
                    break

        return tuple(facts)

    def assess(
        self,
        requirements: QuestionRequirements,
        evidence_by_id: Mapping[str, str],
    ) -> QuestionCoverageAssessment:
        if not isinstance(requirements, QuestionRequirements):
            raise TypeError("Expected QuestionRequirements.")

        facts = self.extract(evidence_by_id)
        coverage: list[RelationshipCoverage] = []

        for relation in requirements.relationships:
            if relation.predicate not in self._SUPPORTED_PREDICATES:
                coverage.append(
                    RelationshipCoverage(
                        relationship_id=relation.relationship_id,
                        status=CoverageStatus.UNKNOWN,
                        supporting_chunk_ids=(),
                        bindings=(),
                    )
                )
                continue

            positive = [
                fact for fact in facts
                if fact.predicate == relation.predicate
                and not fact.negated
                and self._compatible(relation, fact)
            ]

            negative = [
                fact for fact in facts
                if fact.predicate == relation.predicate
                and fact.negated
                and self._compatible(relation, fact)
            ]

            # An explicit positive and negative statement about
            # the same normalized subject-object pair is a conflict.
            conflicting_pair = next(
                (
                    (pos, neg)
                    for pos in positive
                    for neg in negative
                    if _normalize(pos.subject)
                    == _normalize(neg.subject)
                    and _normalize(pos.object)
                    == _normalize(neg.object)
                ),
                None,
            )

            if conflicting_pair is not None:
                ids = tuple(dict.fromkeys(
                    (
                        conflicting_pair[0].chunk_id,
                        conflicting_pair[1].chunk_id,
                    )
                ))

                if len(ids) >= 2:
                    coverage.append(
                        RelationshipCoverage(
                            relationship_id=relation.relationship_id,
                            status=CoverageStatus.CONFLICTING,
                            supporting_chunk_ids=ids,
                            bindings=(),
                        )
                    )
                    continue

                # Same-chunk contradictions cannot currently be
                # represented by the contract requiring >=2 IDs.
                coverage.append(
                    RelationshipCoverage(
                        relationship_id=relation.relationship_id,
                        status=CoverageStatus.UNKNOWN,
                        supporting_chunk_ids=(),
                        bindings=(),
                    )
                )
                continue

            # Multiple distinct matches may represent multiple
            # legitimate answers. Do not arbitrarily choose one.
            if len({
                (
                    _normalize(fact.subject),
                    _normalize(fact.object),
                )
                for fact in positive
            }) != 1:
                coverage.append(
                    RelationshipCoverage(
                        relationship_id=relation.relationship_id,
                        status=(
                            CoverageStatus.MISSING
                            if not positive and not negative
                            else CoverageStatus.UNKNOWN
                        ),
                        supporting_chunk_ids=(),
                        bindings=(),
                    )
                )
                continue

            fact = positive[0]
            bindings = []

            if relation.subject.startswith("?"):
                bindings.append((relation.subject, fact.subject))

            if relation.object.startswith("?"):
                bindings.append((relation.object, fact.object))

            coverage.append(
                RelationshipCoverage(
                    relationship_id=relation.relationship_id,
                    status=CoverageStatus.SUPPORTED,
                    supporting_chunk_ids=tuple(dict.fromkeys(
                        item.chunk_id for item in positive
                    )),
                    bindings=tuple(bindings),
                )
            )

        return QuestionCoverageAssessment(
            requirements=requirements,
            coverage=tuple(coverage),
            assessed_chunk_ids=tuple(evidence_by_id),
            explanation=(
                "Coverage derived from a restricted set of "
                "explicit sentence patterns. Unsupported patterns "
                "and unresolved matches are not semantic proof."
            ),
        )

    @staticmethod
    def _compatible(relation, fact: ExtractedFact) -> bool:
        return (
            (
                relation.subject.startswith("?")
                or _normalize(relation.subject)
                == _normalize(fact.subject)
            )
            and (
                relation.object.startswith("?")
                or _normalize(relation.object)
                == _normalize(fact.object)
            )
        )
