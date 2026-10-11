from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence

from src.diagnosis.runtime import RuntimeDiagnosisResult
from src.recovery.states import RecoveryAction, RecoveryRequest, RecoveryResult, RecoveryStatus
from src.retrieval.reranker import RerankedRetrievalResult


class ControllerDecision(str, Enum):
    KEEP_ORIGINAL = "keep_original"
    ATTEMPT_RECOVERY = "attempt_recovery"


class ControllerOutcome(str, Enum):
    ORIGINAL_ACCEPTED = "original_accepted"
    RECOVERY_PENDING_VERIFICATION = "recovery_pending_verification"
    RECOVERY_FALLBACK = "recovery_fallback"


@dataclass(frozen=True)
class ControllerResult:
    decision: ControllerDecision
    outcome: ControllerOutcome
    diagnosis: RuntimeDiagnosisResult
    original_results: tuple[RerankedRetrievalResult, ...]
    final_results: tuple[RerankedRetrievalResult, ...]
    recovery_result: RecoveryResult | None = None
    reason: str = ""


RecoveryExecutor = Callable[[RecoveryRequest], RecoveryResult]


class RecoveryController:
    """Single-attempt routing controller; no gold annotations or oracle decisions.

    Phase 10.5A does not accept recovered evidence as verified. Successful
    recovery is returned as pending verification, while final_results stays
    at the original evidence until a separate verification gate is added.
    """

    def __init__(
        self,
        recovery_executor: RecoveryExecutor,
        action: RecoveryAction = RecoveryAction.MULTI_QUERY_RETRIEVAL,
    ) -> None:
        if action == RecoveryAction.NONE:
            raise ValueError("A recovery controller needs a non-NONE recovery action.")
        self.recovery_executor = recovery_executor
        self.action = action

    def run(
        self,
        query: str,
        original_results: Sequence[RerankedRetrievalResult],
        diagnosis: RuntimeDiagnosisResult,
    ) -> ControllerResult:
        original = tuple(original_results)
        if not diagnosis.diagnosis.recovery_recommended:
            return ControllerResult(
                decision=ControllerDecision.KEEP_ORIGINAL,
                outcome=ControllerOutcome.ORIGINAL_ACCEPTED,
                diagnosis=diagnosis,
                original_results=original,
                final_results=original,
                reason="Runtime diagnosis did not recommend recovery.",
            )

        request = RecoveryRequest(
            original_query=query,
            action=self.action,
            original_results=original,
        )
        try:
            result = self.recovery_executor(request)
            if result.action != request.action or result.original_query != query or result.original_results != original:
                raise ValueError("Recovery executor returned mismatched request provenance.")
            if result.status == RecoveryStatus.COMPLETED and result.recovered_results:
                return ControllerResult(
                    decision=ControllerDecision.ATTEMPT_RECOVERY,
                    outcome=ControllerOutcome.RECOVERY_PENDING_VERIFICATION,
                    diagnosis=diagnosis,
                    original_results=original,
                    final_results=original,
                    recovery_result=result,
                    reason="Recovered evidence awaits independent verification.",
                )
            reason = "Recovery returned no usable results or did not complete."
        except Exception as exc:
            result = RecoveryResult(
                action=request.action,
                status=RecoveryStatus.FAILED,
                original_query=query,
                recovery_queries=(),
                original_results=original,
                recovered_results=(),
                error_message=f"{type(exc).__name__}: {exc}",
            )
            reason = "Recovery executor failed; original evidence preserved."

        return ControllerResult(
            decision=ControllerDecision.ATTEMPT_RECOVERY,
            outcome=ControllerOutcome.RECOVERY_FALLBACK,
            diagnosis=diagnosis,
            original_results=original,
            final_results=original,
            recovery_result=result,
            reason=reason,
        )
