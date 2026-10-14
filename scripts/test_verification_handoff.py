
"""Phase 11.1B — Verification handoff integration tests."""

from dataclasses import replace
from types import SimpleNamespace

from src.recovery.evidence_selection import (
    EvidencePreservingSelector,
    EvidenceSource,
)
from src.recovery.recovery_handoff import RecoveryHandoff
from src.recovery.verification import (
    VerificationDecision,
    VerificationStatus,
)
from src.recovery.verification_handoff import (
    VerificationHandoffAdapter,
)


def make_result(chunk_id: str, rank: int):
    """Create a lightweight retrieval result for testing."""
    return SimpleNamespace(
        rank=rank,
        chunk=SimpleNamespace(
            id=chunk_id,
            document_id=f"doc_{chunk_id}",
            text=f"Evidence {chunk_id}",
        ),
    )


def make_pending_handoff() -> RecoveryHandoff:
    """Construct a valid recovery handoff with overlapping evidence."""

    original = (
        make_result("a", 1),
        make_result("b", 2),
    )

    recovered = (
        make_result("b", 1),
        make_result("c", 2),
    )

    selection = EvidencePreservingSelector().select(
        original,
        recovered,
    )

    return RecoveryHandoff(
        original_results=original,
        selection=selection,
        pending_verification=True,
        reason="Candidate awaits verification.",
    )


def expect_value_error(callback) -> None:
    try:
        callback()
    except ValueError:
        return

    raise AssertionError("Expected ValueError.")


def test_valid_pending_handoff() -> None:
    handoff = make_pending_handoff()

    result = VerificationHandoffAdapter().prepare(handoff)

    assert result.status == VerificationStatus.NOT_EVALUATED
    assert result.decision == VerificationDecision.UNCERTAIN
    assert not result.accepted

    assert result.original_results == handoff.original_results
    assert result.candidate_results == (
        handoff.selection.selected_results
    )

    assert result.metadata["pending_verification"] is True


def test_nonpending_handoff() -> None:
    handoff = RecoveryHandoff(
        original_results=(make_result("a", 1),),
        selection=None,
        pending_verification=False,
        reason="No recovery required.",
    )

    result = VerificationHandoffAdapter().prepare(handoff)

    assert result.status == VerificationStatus.NOT_EVALUATED
    assert result.decision == VerificationDecision.UNCERTAIN
    assert result.candidate_results == ()
    assert not result.accepted

    assert result.metadata["pending_verification"] is False


def test_missing_selection_rejected() -> None:
    handoff = replace(
        make_pending_handoff(),
        selection=None,
    )

    expect_value_error(
        lambda: VerificationHandoffAdapter().prepare(handoff)
    )


def test_mismatched_original_rejected() -> None:
    handoff = make_pending_handoff()

    bad_selection = replace(
        handoff.selection,
        original_results=(make_result("wrong", 1),),
    )

    expect_value_error(
        lambda: VerificationHandoffAdapter().prepare(
            replace(handoff, selection=bad_selection)
        )
    )


def test_duplicate_candidate_rejected() -> None:
    handoff = make_pending_handoff()
    candidate = handoff.selection.selected_results[0]

    bad_selection = replace(
        handoff.selection,
        selected_results=(candidate, candidate),
    )

    expect_value_error(
        lambda: VerificationHandoffAdapter().prepare(
            replace(handoff, selection=bad_selection)
        )
    )


def test_invalid_source_rejected() -> None:
    handoff = make_pending_handoff()
    candidates = list(handoff.selection.selected_results)

    candidates[0] = replace(
        candidates[0],
        source=EvidenceSource.SECOND_HOP,
    )

    bad_selection = replace(
        handoff.selection,
        selected_results=tuple(candidates),
    )

    expect_value_error(
        lambda: VerificationHandoffAdapter().prepare(
            replace(handoff, selection=bad_selection)
        )
    )


def test_invalid_rank_provenance_rejected() -> None:
    handoff = make_pending_handoff()
    candidates = list(handoff.selection.selected_results)

    candidates[0] = replace(
        candidates[0],
        first_hop_rank=999,
    )

    bad_selection = replace(
        handoff.selection,
        selected_results=tuple(candidates),
    )

    expect_value_error(
        lambda: VerificationHandoffAdapter().prepare(
            replace(handoff, selection=bad_selection)
        )
    )


def main() -> None:
    tests = [
        test_valid_pending_handoff,
        test_nonpending_handoff,
        test_missing_selection_rejected,
        test_mismatched_original_rejected,
        test_duplicate_candidate_rejected,
        test_invalid_source_rejected,
        test_invalid_rank_provenance_rejected,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(
        "\nPHASE 11.1B VERIFICATION HANDOFF TESTS PASSED"
    )


if __name__ == "__main__":
    main()
