
"""Phase 11.4F-5 contradiction-aware path validation tests."""

from src.recovery.contradiction_aware_validator import (
    ContradictionAwarePathValidator,
    PathValidationDecision,
)
from src.recovery.question_requirements import (
    QuestionRequirements,
    RequiredRelationship,
)


VALIDATOR = ContradictionAwarePathValidator()


def rel(rid, subject, predicate, obj):
    return RequiredRelationship(
        relationship_id=rid,
        subject=subject,
        predicate=predicate,
        object=obj,
    )


def req(*relationships, answer="?university"):
    return QuestionRequirements(
        question="Synthetic question",
        relationships=tuple(relationships),
        answer_variable=answer,
    )


def two_hop():
    return req(
        rel("r1", "The Silver Atlas", "authored_by", "?author"),
        rel("r2", "?author", "attended", "?university"),
    )


BOOK = "The Silver Atlas was written by Elena Marsh."
POSITIVE = "Elena Marsh attended Northbridge University."
NEGATIVE = "Elena Marsh never attended Northbridge University."


def test_clean_two_hop():
    report = VALIDATOR.validate(
        two_hop(),
        {"book": BOOK, "education": POSITIVE},
    )

    assert (
        report.decision
        == PathValidationDecision.UNIQUE_UNCONTRADICTED
    )
    assert report.clean_answer_values == (
        "Northbridge University",
    )
    assert not report.contradictions


def test_explicit_contradiction():
    report = VALIDATOR.validate(
        two_hop(),
        {
            "book": BOOK,
            "positive": POSITIVE,
            "negative": NEGATIVE,
        },
    )

    assert report.decision == PathValidationDecision.CONFLICTING
    assert len(report.contradictions) == 1
    assert report.clean_answer_values == ()


def test_same_chunk_contradiction():
    report = VALIDATOR.validate(
        two_hop(),
        {
            "book": BOOK,
            "education": POSITIVE + " " + NEGATIVE,
        },
    )

    assert report.decision == PathValidationDecision.CONFLICTING
    assert len(report.contradictions) == 1
    assert (
        report.contradictions[0].positive_chunk_id
        == report.contradictions[0].negative_chunk_id
    )


def test_irrelevant_negative_fact():
    report = VALIDATOR.validate(
        two_hop(),
        {
            "book": BOOK,
            "education": POSITIVE,
            "negative": (
                "Daniel Reed never attended Lakeside College."
            ),
        },
    )

    assert (
        report.decision
        == PathValidationDecision.UNIQUE_UNCONTRADICTED
    )


def test_disconnected_entity():
    report = VALIDATOR.validate(
        two_hop(),
        {
            "book": BOOK,
            "education": (
                "Daniel Reed attended Lakeside College."
            ),
        },
    )

    assert (
        report.decision
        == PathValidationDecision.NO_COMPLETE_PATH
    )


def test_missing_second_hop():
    report = VALIDATOR.validate(
        two_hop(),
        {"book": BOOK},
    )

    assert (
        report.decision
        == PathValidationDecision.NO_COMPLETE_PATH
    )


def test_negative_only():
    requirements = req(
        rel("r1", "Elena Marsh", "attended", "?university")
    )

    report = VALIDATOR.validate(
        requirements,
        {"negative": NEGATIVE},
    )

    assert (
        report.decision
        == PathValidationDecision.NO_COMPLETE_PATH
    )


def test_multiple_answers():
    requirements = req(
        rel("r1", "Elena Marsh", "attended", "?university")
    )

    report = VALIDATOR.validate(
        requirements,
        {
            "a": POSITIVE,
            "b": "Elena Marsh attended Lakeside College.",
        },
    )

    assert report.decision == PathValidationDecision.AMBIGUOUS
    assert len(report.clean_answer_values) == 2


def test_multiple_answers_one_conflicted():
    requirements = req(
        rel("r1", "Elena Marsh", "attended", "?university")
    )

    report = VALIDATOR.validate(
        requirements,
        {
            "a": POSITIVE,
            "b": "Elena Marsh attended Lakeside College.",
            "negative": NEGATIVE,
        },
    )

    assert report.decision == PathValidationDecision.CONFLICTING
    assert report.clean_answer_values == ("Lakeside College",)


def test_duplicate_positive_evidence():
    requirements = req(
        rel("r1", "Elena Marsh", "attended", "?university")
    )

    report = VALIDATOR.validate(
        requirements,
        {"a": POSITIVE, "b": POSITIVE},
    )

    assert (
        report.decision
        == PathValidationDecision.UNIQUE_UNCONTRADICTED
    )
    assert len(report.paths) == 2


def test_case_insensitive_contradiction():
    requirements = req(
        rel("r1", "Elena Marsh", "attended", "?university")
    )

    report = VALIDATOR.validate(
        requirements,
        {
            "a": POSITIVE,
            "b": (
                "ELENA MARSH never attended "
                "NORTHBRIDGE UNIVERSITY."
            ),
        },
    )

    assert report.decision == PathValidationDecision.CONFLICTING


def test_unsupported_predicate():
    requirements = req(
        rel(
            "r1",
            "Elena Marsh",
            "graduated_from",
            "?university",
        )
    )

    report = VALIDATOR.validate(
        requirements,
        {
            "education": (
                "Elena Marsh graduated from "
                "Northbridge University."
            )
        },
    )

    assert report.decision == PathValidationDecision.UNKNOWN


def test_invalid_requirements():
    try:
        VALIDATOR.validate("invalid", {})
    except TypeError:
        return

    raise AssertionError("Expected TypeError.")


def main():
    tests = [
        test_clean_two_hop,
        test_explicit_contradiction,
        test_same_chunk_contradiction,
        test_irrelevant_negative_fact,
        test_disconnected_entity,
        test_missing_second_hop,
        test_negative_only,
        test_multiple_answers,
        test_multiple_answers_one_conflicted,
        test_duplicate_positive_evidence,
        test_case_insensitive_contradiction,
        test_unsupported_predicate,
        test_invalid_requirements,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(
        "\nPHASE 11.4F-5 CONTRADICTION-AWARE "
        "PATH VALIDATOR TESTS PASSED"
    )
    print(f"Tests passed: {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
