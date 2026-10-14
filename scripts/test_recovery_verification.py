
"""Phase 11.1 tests for verification contracts."""

from src.recovery.verification import (
    VerificationDecision,
    VerificationResult,
    VerificationStatus,
)


def test_unevaluated_candidate() -> None:
    result = VerificationResult(
        status=VerificationStatus.NOT_EVALUATED,
        decision=VerificationDecision.UNCERTAIN,
        reason="Semantic verification has not been performed",
        original_results=("original_chunk",),
        candidate_results=("candidate_chunk",),
    )

    assert not result.accepted
    assert not result.requires_fallback


def test_valid_acceptance() -> None:
    result = VerificationResult(
        status=VerificationStatus.EVALUATED,
        decision=VerificationDecision.ACCEPT,
        reason="Candidate accepted by verification policy",
        original_results=("original_chunk",),
        candidate_results=("candidate_chunk",),
    )

    assert result.accepted


def test_valid_rejection() -> None:
    result = VerificationResult(
        status=VerificationStatus.EVALUATED,
        decision=VerificationDecision.REJECT,
        reason="Candidate rejected by verification policy",
        original_results=("original_chunk",),
        candidate_results=("candidate_chunk",),
    )

    assert result.requires_fallback


def test_unevaluated_acceptance_blocked() -> None:
    try:
        VerificationResult(
            status=VerificationStatus.NOT_EVALUATED,
            decision=VerificationDecision.ACCEPT,
            reason="Invalid premature acceptance",
            original_results=("original_chunk",),
            candidate_results=("candidate_chunk",),
        )
    except ValueError:
        return

    raise AssertionError("Unevaluated acceptance should be rejected")


def test_empty_candidate_acceptance_blocked() -> None:
    try:
        VerificationResult(
            status=VerificationStatus.EVALUATED,
            decision=VerificationDecision.ACCEPT,
            reason="Invalid empty candidate",
            original_results=("original_chunk",),
            candidate_results=(),
        )
    except ValueError:
        return

    raise AssertionError("Empty candidate acceptance should be rejected")


def main() -> None:
    tests = [
        test_unevaluated_candidate,
        test_valid_acceptance,
        test_valid_rejection,
        test_unevaluated_acceptance_blocked,
        test_empty_candidate_acceptance_blocked,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("\nPHASE 11.1 VERIFICATION CONTRACT TESTS PASSED")


if __name__ == "__main__":
    main()
