
"""Gold-free, deterministic post-recovery evidence comparison.

Phase 11.2A measures lexical and structural changes.
It does not establish answer sufficiency or authorize acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.diagnosis.evidence_signals import (
    informative_query_tokens,
    tokenize,
)
from src.recovery.evidence_selection import SelectedEvidence
from src.retrieval.reranker import RerankedRetrievalResult


@dataclass(frozen=True)
class EvidenceSetAssessment:
    chunk_count: int
    unique_document_count: int
    informative_query_token_count: int
    covered_query_tokens: frozenset[str]
    uncovered_query_tokens: frozenset[str]
    query_token_coverage: float | None
    chunk_ids: frozenset[str]


@dataclass(frozen=True)
class EvidenceComparison:
    original: EvidenceSetAssessment
    candidate: EvidenceSetAssessment
    gained_query_tokens: frozenset[str]
    lost_query_tokens: frozenset[str]
    new_document_count: int
    retained_chunk_count: int
    coverage_delta: float | None


class EvidenceAssessmentBaseline:
    """
    Compare original and selected evidence without gold labels.

    Query-token coverage is diagnostic only.
    It is not a reliable measure of multi-hop sufficiency.
    """

    @staticmethod
    def _assess(
        query: str,
        evidence: Sequence[
            RerankedRetrievalResult | SelectedEvidence
        ],
    ) -> EvidenceSetAssessment:
        query_tokens = set(informative_query_tokens(query))

        covered: set[str] = set()
        document_ids: set[str] = set()
        chunk_ids: set[str] = set()

        for item in evidence:
            chunk = item.chunk

            chunk_ids.add(chunk.id)
            document_ids.add(chunk.document_id)

            covered.update(
                query_tokens.intersection(tokenize(chunk.text))
            )

        uncovered = query_tokens - covered

        coverage = (
            len(covered) / len(query_tokens)
            if query_tokens
            else None
        )

        return EvidenceSetAssessment(
            chunk_count=len(evidence),
            unique_document_count=len(document_ids),
            informative_query_token_count=len(query_tokens),
            covered_query_tokens=frozenset(covered),
            uncovered_query_tokens=frozenset(uncovered),
            query_token_coverage=coverage,
            chunk_ids=frozenset(chunk_ids),
        )

    def compare(
        self,
        query: str,
        original_results: Sequence[RerankedRetrievalResult],
        candidate_results: Sequence[SelectedEvidence],
    ) -> EvidenceComparison:
        original = self._assess(query, original_results)
        candidate = self._assess(query, candidate_results)

        original_document_ids = {
            item.chunk.document_id for item in original_results
        }
        candidate_document_ids = {
            item.chunk.document_id for item in candidate_results
        }

        coverage_delta = (
            candidate.query_token_coverage
            - original.query_token_coverage
            if original.query_token_coverage is not None
            and candidate.query_token_coverage is not None
            else None
        )

        return EvidenceComparison(
            original=original,
            candidate=candidate,
            gained_query_tokens=(
                candidate.covered_query_tokens
                - original.covered_query_tokens
            ),
            lost_query_tokens=(
                original.covered_query_tokens
                - candidate.covered_query_tokens
            ),
            new_document_count=len(
                candidate_document_ids - original_document_ids
            ),
            retained_chunk_count=len(
                original.chunk_ids & candidate.chunk_ids
            ),
            coverage_delta=coverage_delta,
        )
