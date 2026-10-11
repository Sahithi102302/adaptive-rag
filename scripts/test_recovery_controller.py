from __future__ import annotations

from types import SimpleNamespace

from src.recovery.controller import ControllerDecision, ControllerOutcome, RecoveryController
from src.recovery.states import RecoveryAction, RecoveryResult, RecoveryStatus


def diagnosis(recommend: bool):
    return SimpleNamespace(diagnosis=SimpleNamespace(recovery_recommended=recommend))


def result(request, status=RecoveryStatus.COMPLETED, recovered=("new",)):
    return RecoveryResult(
        action=request.action,
        status=status,
        original_query=request.original_query,
        recovery_queries=("second hop",),
        original_results=request.original_results,
        recovered_results=tuple(recovered),
    )


def main():
    original = ("first",)
    calls = []

    def executor(request):
        calls.append(request)
        return result(request)

    controller = RecoveryController(executor)
    kept = controller.run("question", original, diagnosis(False))
    assert kept.decision == ControllerDecision.KEEP_ORIGINAL
    assert kept.outcome == ControllerOutcome.ORIGINAL_ACCEPTED
    assert kept.final_results == original
    assert not calls
    print("PASS: sufficient evidence bypasses recovery")

    pending = controller.run("question", original, diagnosis(True))
    assert len(calls) == 1
    assert calls[0].action == RecoveryAction.MULTI_QUERY_RETRIEVAL
    assert pending.outcome == ControllerOutcome.RECOVERY_PENDING_VERIFICATION
    assert pending.final_results == original
    assert pending.recovery_result.recovered_results == ("new",)
    print("PASS: successful recovery waits for verification")

    empty = RecoveryController(lambda request: result(request, recovered=())).run("q", original, diagnosis(True))
    assert empty.outcome == ControllerOutcome.RECOVERY_FALLBACK
    assert empty.final_results == original
    print("PASS: empty recovery falls back")

    failed = RecoveryController(lambda request: result(request, status=RecoveryStatus.FAILED)).run("q", original, diagnosis(True))
    assert failed.outcome == ControllerOutcome.RECOVERY_FALLBACK
    print("PASS: failed recovery falls back")

    def raises(request):
        raise RuntimeError("injected failure")

    raised = RecoveryController(raises).run("q", original, diagnosis(True))
    assert raised.outcome == ControllerOutcome.RECOVERY_FALLBACK
    assert raised.recovery_result.status == RecoveryStatus.FAILED
    assert "injected failure" in raised.recovery_result.error_message
    print("PASS: executor exception is contained")

    def mismatched(request):
        return RecoveryResult(
            action=request.action, status=RecoveryStatus.COMPLETED,
            original_query="wrong question", recovery_queries=("q",),
            original_results=request.original_results, recovered_results=("new",),
        )

    mismatch = RecoveryController(mismatched).run("q", original, diagnosis(True))
    assert mismatch.outcome == ControllerOutcome.RECOVERY_FALLBACK
    assert mismatch.final_results == original
    print("PASS: mismatched provenance is rejected")

    print("PHASE 10.5A CONTROLLER TESTS PASSED")


if __name__ == "__main__":
    main()
