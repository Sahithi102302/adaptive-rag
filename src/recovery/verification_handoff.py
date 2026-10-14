
"""Bridge Phase 10 recovery handoffs to Phase 11 verification contracts."""

from __future__ import annotations

from src.recovery.evidence_selection import EvidenceSource
from src.recovery.recovery_handoff import RecoveryHandoff
from src.recovery.verification import (
    VerificationDecision,
    VerificationResult,
    VerificationStatus,
)


class VerificationHandoffAdapter:
    """
    Convert a valid recovery handoff into an unevaluated
    verification request.

    This adapter never accepts recovered evidence.
    """

    def prepare(self, handoff: RecoveryHandoff) -> VerificationResult:
        if not isinstance(handoff, RecoveryHandoff):
            raise TypeError("Expected a RecoveryHandoff.")

        original = handoff.original_results

        if not isinstance(original, tuple):
            raise ValueError("Original evidence must be a tuple.")

        if not handoff.pending_verification:
            if handoff.selection is not None:
                raise ValueError(
                    "Nonpending handoff unexpectedly contains a candidate."
                )

            return VerificationResult(
                status=VerificationStatus.NOT_EVALUATED,
                decision=VerificationDecision.UNCERTAIN,
                reason="No recovery candidate awaits verification.",
                original_results=original,
                candidate_results=(),
                metadata={"pending_verification": False},
            )

        selection = handoff.selection

        if selection is None:
            raise ValueError(
                "Pending verification requires an evidence selection."
            )

        if selection.original_results != original:
            raise ValueError(
                "Candidate original evidence does not match the handoff."
            )

        if not selection.recovered_results:
            raise ValueError(
                "Pending recovery must contain recovered evidence."
            )

        candidates = selection.selected_results

        if not candidates:
            raise ValueError("Selected candidate evidence is empty.")

        original_ids = {
            result.chunk.id for result in original
        }
        recovered_ids = {
            result.chunk.id for result in selection.recovered_results
        }

        seen_ids: set[str] = set()

        for expected_rank, item in enumerate(candidates, start=1):
            if item.rank != expected_rank:
                raise ValueError(
                    "Candidate ranks must be consecutive, starting at 1."
                )

            if item.chunk_id != item.chunk.id:
                raise ValueError(
                    "Candidate chunk ID does not match its chunk."
                )

            if item.chunk_id in seen_ids:
                raise ValueError(
                    "Candidate selection contains duplicate chunk IDs."
                )

            seen_ids.add(item.chunk_id)

            in_first = item.chunk_id in original_ids
            in_second = item.chunk_id in recovered_ids

            if not (in_first or in_second):
                raise ValueError(
                    "Candidate contains evidence from neither retrieval hop."
                )

            expected_source = (
                EvidenceSource.BOTH
                if in_first and in_second
                else EvidenceSource.FIRST_HOP
                if in_first
                else EvidenceSource.SECOND_HOP
            )

            if item.source != expected_source:
                raise ValueError(
                    "Candidate source provenance is inconsistent."
                )

            first_matches = [
                result.rank
                for result in original
                if result.chunk.id == item.chunk_id
            ]
            second_matches = [
                result.rank
                for result in selection.recovered_results
                if result.chunk.id == item.chunk_id
            ]

            expected_first_rank = (
                first_matches[0] if first_matches else None
            )
            expected_second_rank = (
                second_matches[0] if second_matches else None
            )

            if (
                item.first_hop_rank != expected_first_rank
                or item.second_hop_rank != expected_second_rank
            ):
                raise ValueError(
                    "Candidate retrieval-rank provenance is inconsistent."
                )

        return VerificationResult(
            status=VerificationStatus.NOT_EVALUATED,
            decision=VerificationDecision.UNCERTAIN,
            reason=(
                "Recovery candidate prepared with valid provenance; "
                "semantic sufficiency has not been evaluated."
            ),
            original_results=original,
            candidate_results=tuple(candidates),
            metadata={
                "pending_verification": True,
                "candidate_count": len(candidates),
                "original_count": len(original),
                "recovered_count": len(selection.recovered_results),
            },
        )
