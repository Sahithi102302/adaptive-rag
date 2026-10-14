
"""Tests for the Phase 11.4B conservative semantic assessor."""

from dataclasses import dataclass

from src.recovery.semantic_assessor import ConservativeSemanticAssessor
from src.recovery.semantic_verification import SemanticSupport


@dataclass(frozen=True)
class FakeChunk:
    id: str
    text: str


@dataclass(frozen=True)
class FakeResult:
    chunk: FakeChunk


def evidence(chunk_id, text):
    return FakeResult(FakeChunk(chunk_id, text))


def expect_error(callback, error_type):
    try:
        callback()
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__}.")


def test_empty_evidence_is_unknown():
    result = ConservativeSemanticAssessor().assess(
        "Who wrote Hamlet?", ()
    )
    assert result.support == SemanticSupport.UNKNOWN


def test_lexical_overlap_is_not_full_support():
    result = ConservativeSemanticAssessor().assess(
        "Who wrote Hamlet?",
        (evidence("a", "Hamlet was written by William Shakespeare."),),
    )
    assert result.support != SemanticSupport.SUPPORTED
    assert result.assessed_chunk_ids == ("a",)


def test_unrelated_evidence_is_unsupported():
    result = ConservativeSemanticAssessor().assess(
        "Who wrote Hamlet?",
        (evidence("a", "The Pacific Ocean is very large."),),
    )
    assert result.support == SemanticSupport.UNSUPPORTED


def test_missing_question_terms_reported():
    result = ConservativeSemanticAssessor().assess(
        "Which university did Ada Lovelace attend?",
        (evidence("a", "Ada Lovelace was a mathematician."),),
    )
    assert result.has_missing_link


def test_duplicate_chunk_rejected():
    expect_error(
        lambda: ConservativeSemanticAssessor().assess(
            "Who wrote Hamlet?",
            (
                evidence("a", "Hamlet is a play."),
                evidence("a", "Shakespeare was a writer."),
            ),
        ),
        ValueError,
    )


def test_empty_question_rejected():
    expect_error(
        lambda: ConservativeSemanticAssessor().assess(
            " ", (evidence("a", "Some text."),)
        ),
        ValueError,
    )


def main():
    tests = [
        test_empty_evidence_is_unknown,
        test_lexical_overlap_is_not_full_support,
        test_unrelated_evidence_is_unsupported,
        test_missing_question_terms_reported,
        test_duplicate_chunk_rejected,
        test_empty_question_rejected,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("\nPHASE 11.4B CONSERVATIVE ASSESSOR TESTS PASSED")


if __name__ == "__main__":
    main()
