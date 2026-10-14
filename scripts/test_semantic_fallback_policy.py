
"""Phase 11.4G-12F: semantic fallback policy regression tests.

Tests conservative fallback screening behavior.

Important:
- CANDIDATE means further review only.
- These tests do not establish semantic correctness.
- Passing tests do not authorize fallback acceptance.
"""

from __future__ import annotations

from src.recovery.semantic_fallback_policy import (
    FallbackEligibility,
    SemanticFallbackPolicy,
)


policy = SemanticFallbackPolicy()


def test_novelist_candidate():
    question = (
        "Which university did the novelist behind "
        "The Copper Horizon attend?"
    )

    result = policy.screen(question)

    assert result.eligibility == FallbackEligibility.CANDIDATE


def test_educated_candidate():
    question = (
        "At what college was the author of "
        "The Last Meridian educated?"
    )

    result = policy.screen(question)

    assert result.eligibility == FallbackEligibility.CANDIDATE


def test_authored_candidate():
    question = (
        "Where did the person who authored "
        "The Paper Constellation go to college?"
    )

    result = policy.screen(question)

    assert result.eligibility == FallbackEligibility.CANDIDATE


def test_employment_blocked():
    question = (
        "Which university employed the author "
        "of The Copper Horizon?"
    )

    result = policy.screen(question)

    assert result.eligibility == FallbackEligibility.BLOCKED
    assert result.reason == "unsupported_relation"


def test_coauthor_blocked():
    question = (
        "Which university did the co-author of "
        "The Crimson Observatory attend?"
    )

    result = policy.screen(question)

    assert result.eligibility == FallbackEligibility.BLOCKED
    assert result.reason == "unsupported_role"


def test_uncertainty_blocked():
    question = (
        "Which university might the author of "
        "The Paper Constellation have attended?"
    )

    result = policy.screen(question)

    assert result.eligibility == FallbackEligibility.BLOCKED
    assert result.reason == "uncertainty_or_hypothetical"


def test_additional_constraint_blocked():
    question = (
        "Which university did the author of "
        "The Copper Horizon attend first?"
    )

    result = policy.screen(question)

    assert result.eligibility == FallbackEligibility.BLOCKED
    assert result.reason == "additional_constraint"


def test_compound_blocked():
    question = (
        "Which university did the author of "
        "The Last Meridian attend, and what degree "
        "did they earn?"
    )

    result = policy.screen(question)

    assert result.eligibility == FallbackEligibility.BLOCKED
    assert result.reason == "compound_request"


def test_one_hop_unresolved():
    question = "Who authored The Paper Constellation?"

    result = policy.screen(question)

    assert result.eligibility == FallbackEligibility.UNRESOLVED
    assert result.reason == "missing_task_cues"


def test_empty_blocked():
    result = policy.screen("")

    assert result.eligibility == FallbackEligibility.BLOCKED
    assert result.reason == "empty_question"


def test_temporal_constraint_blocked():
    """Regression test for the A25 temporal-constraint failure."""

    question = (
        "Which university did the author of "
        "The Last Meridian attend in 1998?"
    )

    result = policy.screen(question)

    assert result.eligibility == FallbackEligibility.BLOCKED
    assert result.reason == "temporal_constraint"


def main():
    tests = [
        test_novelist_candidate,
        test_educated_candidate,
        test_authored_candidate,
        test_employment_blocked,
        test_coauthor_blocked,
        test_uncertainty_blocked,
        test_additional_constraint_blocked,
        test_compound_blocked,
        test_one_hop_unresolved,
        test_empty_blocked,
        test_temporal_constraint_blocked,
    ]

    passed = 0

    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
            passed += 1

        except AssertionError as exc:
            print(f"FAIL: {test.__name__}")
            raise AssertionError(
                f"Test failed: {test.__name__}"
            ) from exc

    print()
    print("PHASE 11.4G-12F FALLBACK POLICY TESTS PASSED")
    print(f"Tests passed: {passed}/{len(tests)}")


if __name__ == "__main__":
    main()
