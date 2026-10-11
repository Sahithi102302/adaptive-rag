from __future__ import annotations

from dataclasses import dataclass

from src.recovery.controller import ControllerOutcome, ControllerResult
from src.recovery.evidence_selection import (
    EvidencePreservingSelector,
    EvidenceSelectionResult,
)
from src.retrieval.reranker import RerankedRetrievalResult


@dataclass(frozen=True)
class RecoveryHandoff:
    """Candidate preparation only; no verification or acceptance decision."""

    original_results: tuple[RerankedRetrievalResult, ...]
    selection: EvidenceSelectionResult | None
    pending_verification: bool
    reason: str


class RecoveryHandoffBuilder:
    def __init__(self, selector: EvidencePreservingSelector | None = None) -> None:
        self.selector = selector or EvidencePreservingSelector()

    def build(self, controller_result: ControllerResult) -> RecoveryHandoff:
        original = controller_result.original_results
        if controller_result.final_results != original:
            raise ValueError("Controller changed original evidence before verification.")

        if controller_result.outcome != ControllerOutcome.RECOVERY_PENDING_VERIFICATION:
            return RecoveryHandoff(
                original_results=original,
                selection=None,
                pending_verification=False,
                reason="No recovery candidate awaits verification.",
            )

        recovery = controller_result.recovery_result
        if recovery is None or not recovery.succeeded or not recovery.recovered_results:
            raise ValueError("Pending recovery has no completed, nonempty recovery result.")
        if recovery.original_results != original:
            raise ValueError("Recovery candidate has mismatched original evidence provenance.")

        selection = self.selector.select(original, recovery.recovered_results)
        if not selection.selected_results:
            raise ValueError("Candidate selection produced no evidence.")

        return RecoveryHandoff(
            original_results=original,
            selection=selection,
            pending_verification=True,
            reason="Candidate evidence prepared; not verified or accepted.",
        )
