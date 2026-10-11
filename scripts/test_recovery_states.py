from src.recovery.states import (
    RecoveryAction,
    RecoveryRequest,
    RecoveryResult,
    RecoveryStatus,
)


def main() -> None:

    print("=" * 100)
    print(
        "PHASE 10.1 — RECOVERY STATE MODEL TEST"
    )
    print("=" * 100)

    request = RecoveryRequest(
        original_query=(
            "Which writer created the work?"
        ),
        action=(
            RecoveryAction.EXPANDED_CANDIDATE_RETRIEVAL
        ),
        original_results=(),
        metadata={
            "source": "phase_10_1_test",
        },
    )

    assert (
        request.action
        == RecoveryAction.EXPANDED_CANDIDATE_RETRIEVAL
    )

    assert request.original_results == ()

    completed = RecoveryResult(
        action=request.action,
        status=RecoveryStatus.COMPLETED,
        original_query=request.original_query,
        recovery_queries=(
            request.original_query,
        ),
        original_results=request.original_results,
        recovered_results=(),
        metadata={
            "candidate_k": 40,
        },
    )

    assert completed.attempted is True
    assert completed.succeeded is True

    not_attempted = RecoveryResult(
        action=RecoveryAction.NONE,
        status=RecoveryStatus.NOT_ATTEMPTED,
        original_query=request.original_query,
        recovery_queries=(),
        original_results=request.original_results,
        recovered_results=request.original_results,
    )

    assert not_attempted.attempted is False
    assert not_attempted.succeeded is False

    failed = RecoveryResult(
        action=request.action,
        status=RecoveryStatus.FAILED,
        original_query=request.original_query,
        recovery_queries=(
            request.original_query,
        ),
        original_results=request.original_results,
        recovered_results=(),
        error_message=(
            "Synthetic Phase 10.1 test failure."
        ),
    )

    assert failed.attempted is True
    assert failed.succeeded is False

    print()
    print(
        "Recovery actions:"
    )

    for action in RecoveryAction:
        print(
            f"  - {action.value}"
        )

    print()
    print(
        "Recovery statuses:"
    )

    for status in RecoveryStatus:
        print(
            f"  - {status.value}"
        )

    print()
    print(
        "Completed recovery:"
    )
    print(
        f"  attempted: {completed.attempted}"
    )
    print(
        f"  succeeded: {completed.succeeded}"
    )

    print()
    print(
        "Not-attempted recovery:"
    )
    print(
        f"  attempted: {not_attempted.attempted}"
    )
    print(
        f"  succeeded: {not_attempted.succeeded}"
    )

    print()
    print(
        "Failed recovery:"
    )
    print(
        f"  attempted: {failed.attempted}"
    )
    print(
        f"  succeeded: {failed.succeeded}"
    )

    print()
    print("=" * 100)
    print(
        "PHASE 10.1 RECOVERY STATE MODEL TEST PASSED"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()