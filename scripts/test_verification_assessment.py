
"""Phase 11.2B verification assessment integration tests."""

from dataclasses import replace

from src.recovery.verification import (
    VerificationDecision,
    VerificationStatus,
)
from src.recovery.verification_assessment import (
    VerificationAssessmentCoordinator,
)

from scripts.test_verification_handoff import (
    make_pending_handoff,
)
from src.recovery.verification_handoff import (
    VerificationHandoffAdapter,
)


def make_prepared():
    handoff = make_pending_handoff()
    return VerificationHandoffAdapter().prepare(handoff)


def expect_value_error(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("Expected ValueError.")


def test_diagnostics_attached():
    prepared = make_prepared()

    assessed = VerificationAssessmentCoordinator().assess(
        query="Which evidence explains a?",
        prepared=prepared,
    )

    assert "evidence_assessment" in assessed.metadata
    assert assessed.metadata["assessment_version"] == (
        "phase_11_2b_lexical"
    )


def test_candidate_not_accepted():
    prepared = make_prepared()

    assessed = VerificationAssessmentCoordinator().assess(
        query="Which evidence explains a?",
        prepared=prepared,
    )

    assert assessed.status == VerificationStatus.NOT_EVALUATED
    assert assessed.decision == VerificationDecision.UNCERTAIN
    assert not assessed.accepted


def test_original_evidence_preserved():
    prepared = make_prepared()

    assessed = VerificationAssessmentCoordinator().assess(
        query="Which evidence explains a?",
        prepared=prepared,
    )

    assert assessed.original_results == prepared.original_results
    assert assessed.candidate_results == prepared.candidate_results


def test_no_candidate_pass_through():
    prepared = VerificationHandoffAdapter().prepare(
        replace(
            make_pending_handoff(),
            pending_verification=False,
            selection=None,
        )
    )

    assessed = VerificationAssessmentCoordinator().assess(
        query="What happened?",
        prepared=prepared,
    )

    assert assessed is prepared
    assert assessed.candidate_results == ()


def test_empty_query_rejected():
    prepared = make_prepared()

    expect_value_error(
        lambda: VerificationAssessmentCoordinator().assess(
            query=" ",
            prepared=prepared,
        )
    )


def test_already_evaluated_rejected():
    prepared = make_prepared()

    evaluated = replace(
        prepared,
        status=VerificationStatus.EVALUATED,
        decision=VerificationDecision.REJECT,
        reason="Rejected by a separate verifier.",
    )

    expect_value_error(
        lambda: VerificationAssessmentCoordinator().assess(
            query="What happened?",
            prepared=evaluated,
        )
    )


def main():
    tests = [
        test_diagnostics_attached,
        test_candidate_not_accepted,
        test_original_evidence_preserved,
        test_no_candidate_pass_through,
        test_empty_query_rejected,
        test_already_evaluated_rejected,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("\nPHASE 11.2B VERIFICATION ASSESSMENT TESTS PASSED")


if __name__ == "__main__":
    main()
