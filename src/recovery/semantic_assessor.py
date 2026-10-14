
"""Conservative question-aware semantic assessment baseline.

Phase 11.4B provides a deterministic reference assessor.
It does not establish semantic entailment or answer sufficiency.
"""

from __future__ import annotations

from typing import Sequence

from src.diagnosis.evidence_signals import informative_query_tokens, tokenize
from src.recovery.semantic_verification import (
    SemanticEvidenceAssessment,
    SemanticFinding,
    SemanticFindingType,
    SemanticSupport,
)


class ConservativeSemanticAssessor:
    """Produce provenance-grounded diagnostic findings without an LLM.

    Lexical overlap can identify potentially relevant evidence but cannot
    justify a SUPPORTED decision. Therefore this baseline only returns
    PARTIAL, UNSUPPORTED, or UNKNOWN.
    """

    name = "conservative_lexical_baseline_v1"

    def assess(self, question: str, evidence: Sequence) -> SemanticEvidenceAssessment:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a nonempty string.")

        assessed_ids = []
        evidence_tokens = set()
        cited_ids = []

        for item in evidence:
            chunk = item.chunk
            chunk_id = chunk.id

            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError("Evidence chunk ID must be nonempty.")

            if chunk_id in assessed_ids:
                raise ValueError("Duplicate evidence chunk ID.")

            if not isinstance(chunk.text, str):
                raise TypeError("Evidence chunk text must be a string.")

            assessed_ids.append(chunk_id)

            tokens = set(tokenize(chunk.text))
            evidence_tokens.update(tokens)

            if tokens.intersection(informative_query_tokens(question)):
                cited_ids.append(chunk_id)

        query_tokens = set(informative_query_tokens(question))

        if not evidence:
            return SemanticEvidenceAssessment(
                support=SemanticSupport.UNKNOWN,
                findings=(),
                assessed_chunk_ids=(),
                explanation="No evidence available for assessment.",
                assessor_name=self.name,
            )

        if not query_tokens:
            return SemanticEvidenceAssessment(
                support=SemanticSupport.UNKNOWN,
                findings=(),
                assessed_chunk_ids=tuple(assessed_ids),
                explanation="Question has no informative lexical tokens.",
                assessor_name=self.name,
            )

        covered = query_tokens.intersection(evidence_tokens)
        missing = query_tokens - covered

        findings = []

        if covered:
            findings.append(
                SemanticFinding(
                    finding_type=SemanticFindingType.SUPPORT,
                    explanation=(
                        "Cited chunks contain informative question tokens; "
                        "semantic support has not been established."
                    ),
                    supporting_chunk_ids=tuple(cited_ids),
                )
            )

        if missing:
            findings.append(
                SemanticFinding(
                    finding_type=SemanticFindingType.MISSING_LINK,
                    explanation=(
                        "Some informative question tokens are absent; "
                        "this is a lexical gap, not proof of a missing reasoning link."
                    ),
                    supporting_chunk_ids=(),
                )
            )

        support = (
            SemanticSupport.PARTIAL
            if covered
            else SemanticSupport.UNSUPPORTED
        )

        return SemanticEvidenceAssessment(
            support=support,
            findings=tuple(findings),
            assessed_chunk_ids=tuple(assessed_ids),
            explanation=(
                "Conservative lexical diagnostic only; "
                "semantic entailment and multi-hop sufficiency remain unverified."
            ),
            assessor_name=self.name,
        )
