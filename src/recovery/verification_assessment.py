
"""Attach gold-free evidence diagnostics to a verification request.

Phase 11.2B is diagnostic only. It cannot authorize ACCEPT.
"""

from __future__ import annotations

from dataclasses import asdict, replace

from src.recovery.evidence_assessment import EvidenceAssessmentBaseline
from src.recovery.verification import (
    VerificationDecision,
    VerificationResult,
    VerificationStatus,
)


class VerificationAssessmentCoordinator:
    """Assess prepared recovery candidates without accepting them."""

    def __init__(
        self,
        assessor: EvidenceAssessmentBaseline | None = None,
    ) -> None:
        self.assessor = assessor or EvidenceAssessmentBaseline()

    def assess(
        self,
        query: str,
        prepared: VerificationResult,
    ) -> VerificationResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Verification query must be nonempty.")

        if not isinstance(prepared, VerificationResult):
            raise TypeError("Expected a VerificationResult.")

        if (
            prepared.status != VerificationStatus.NOT_EVALUATED
            or prepared.decision != VerificationDecision.UNCERTAIN
        ):
            raise ValueError(
                "Only unevaluated verification requests can be assessed."
            )

        metadata = dict(prepared.metadata or {})

        if not metadata.get("pending_verification", False):
            if prepared.candidate_results:
                raise ValueError(
                    "Nonpending request unexpectedly contains candidates."
                )
            return prepared

        if not prepared.candidate_results:
            raise ValueError(
                "Pending verification requires candidate evidence."
            )

        comparison = self.assessor.compare(
            query=query,
            original_results=prepared.original_results,
            candidate_results=prepared.candidate_results,
        )

        diagnostics = {
            "original_chunk_count": comparison.original.chunk_count,
            "candidate_chunk_count": comparison.candidate.chunk_count,
            "original_document_count": (
                comparison.original.unique_document_count
            ),
            "candidate_document_count": (
                comparison.candidate.unique_document_count
            ),
            "original_query_token_coverage": (
                comparison.original.query_token_coverage
            ),
            "candidate_query_token_coverage": (
                comparison.candidate.query_token_coverage
            ),
            "coverage_delta": comparison.coverage_delta,
            "gained_query_tokens": sorted(
                comparison.gained_query_tokens
            ),
            "lost_query_tokens": sorted(
                comparison.lost_query_tokens
            ),
            "new_document_count": comparison.new_document_count,
            "retained_chunk_count": comparison.retained_chunk_count,
        }

        metadata["evidence_assessment"] = diagnostics
        metadata["assessment_version"] = "phase_11_2b_lexical"
        metadata["semantic_sufficiency_verified"] = False

        return replace(
            prepared,
            status=VerificationStatus.NOT_EVALUATED,
            decision=VerificationDecision.UNCERTAIN,
            reason=(
                "Gold-free lexical and structural evidence comparison "
                "completed; answer sufficiency remains unverified."
            ),
            metadata=metadata,
        )
