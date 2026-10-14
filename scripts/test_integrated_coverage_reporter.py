
"""Phase 11.4F-6 integrated coverage reporting tests."""

from src.recovery.integrated_coverage_reporter import (
    IntegratedCoverageReporter,
    IntegratedDecision,
)
from src.recovery.question_requirements import (
    QuestionRequirements,
    RequiredRelationship,
)


REPORTER = IntegratedCoverageReporter()

BOOK = "The Silver Atlas was written by Elena Marsh."
EDUCATION = "Elena Marsh attended Northbridge University."
NEGATIVE = "Elena Marsh never attended Northbridge University."


def relation(rid, subject, predicate, obj):
    return RequiredRelationship(
        relationship_id=rid,
        subject=subject,
        predicate=predicate,
        object=obj,
    )


def requirements(*relationships, answer="?university"):
    return QuestionRequirements(
        question="Which university did the author attend?",
        relationships=tuple(relationships),
        answer_variable=answer,
    )


def two_hop():
    return requirements(
        relation(
            "r1", "The Silver Atlas", "authored_by", "?author"
        ),
        relation(
            "r2", "?author", "attended", "?university"
        ),
    )


def test_complete_path():
    report = REPORTER.assess(
        two_hop(),
        {"book": BOOK, "education": EDUCATION},
    )

    assert report.decision == IntegratedDecision.UNIQUE_CANDIDATE
    assert report.candidate_answers == ("Northbridge University",)
    assert len(report.path_summaries) == 1
    assert report.path_summaries[0].supporting_chunk_ids == (
        "book", "education"
    )
    assert not report.eligible_for_automatic_acceptance


def test_irrelevant_distractor():
    report = REPORTER.assess(
        two_hop(),
        {
            "book": BOOK,
            "education": EDUCATION,
            "distractor": "Daniel Reed attended Lakeside College.",
        },
    )

    assert report.decision == IntegratedDecision.UNIQUE_CANDIDATE
    assert report.candidate_answers == ("Northbridge University",)


def test_disconnected_not_contradictory():
    report = REPORTER.assess(
        two_hop(),
        {
            "book": BOOK,
            "education": "Daniel Reed attended Lakeside College.",
        },
    )

    assert report.decision == IntegratedDecision.INCOMPLETE
    assert report.contradictions == ()


def test_missing_second_hop():
    report = REPORTER.assess(two_hop(), {"book": BOOK})

    assert report.decision == IntegratedDecision.INCOMPLETE
    assert report.path_summaries == ()


def test_missing_first_hop():
    report = REPORTER.assess(
        two_hop(), {"education": EDUCATION}
    )

    assert report.decision == IntegratedDecision.INCOMPLETE


def test_explicit_contradiction():
    report = REPORTER.assess(
        two_hop(),
        {
            "book": BOOK,
            "positive": EDUCATION,
            "negative": NEGATIVE,
        },
    )

    assert report.decision == IntegratedDecision.CONFLICTING
    assert len(report.contradictions) == 1
    assert len(report.path_summaries) == 1
    assert report.path_summaries[0].contradictions


def test_same_chunk_contradiction():
    report = REPORTER.assess(
        two_hop(),
        {
            "book": BOOK,
            "education": EDUCATION + " " + NEGATIVE,
        },
    )

    assert report.decision == IntegratedDecision.CONFLICTING
    assert len(report.contradictions) == 1


def test_multiple_answers():
    req = requirements(
        relation(
            "r1", "Elena Marsh", "attended", "?university"
        )
    )

    report = REPORTER.assess(
        req,
        {
            "a": EDUCATION,
            "b": "Elena Marsh attended Lakeside College.",
        },
    )

    assert report.decision == IntegratedDecision.AMBIGUOUS
    assert len(report.candidate_answers) == 2


def test_conflict_with_clean_alternative():
    req = requirements(
        relation(
            "r1", "Elena Marsh", "attended", "?university"
        )
    )

    report = REPORTER.assess(
        req,
        {
            "a": EDUCATION,
            "b": "Elena Marsh attended Lakeside College.",
            "negative": NEGATIVE,
        },
    )

    assert report.decision == IntegratedDecision.CONFLICTING
    assert report.clean_candidate_answers == ("Lakeside College",)


def test_unknown_predicate():
    req = requirements(
        relation(
            "r1", "Elena Marsh",
            "graduated_from", "?university",
        )
    )

    report = REPORTER.assess(
        req,
        {
            "education": (
                "Elena Marsh graduated from Northbridge University."
            )
        },
    )

    assert report.decision == IntegratedDecision.UNKNOWN


def test_no_automatic_acceptance():
    report = REPORTER.assess(
        two_hop(),
        {"book": BOOK, "education": EDUCATION},
    )

    assert report.decision == IntegratedDecision.UNIQUE_CANDIDATE
    assert report.eligible_for_automatic_acceptance is False


def test_invalid_requirements():
    try:
        REPORTER.assess("invalid", {})
    except TypeError:
        return

    raise AssertionError("Expected TypeError.")


def main():
    tests = [
        test_complete_path,
        test_irrelevant_distractor,
        test_disconnected_not_contradictory,
        test_missing_second_hop,
        test_missing_first_hop,
        test_explicit_contradiction,
        test_same_chunk_contradiction,
        test_multiple_answers,
        test_conflict_with_clean_alternative,
        test_unknown_predicate,
        test_no_automatic_acceptance,
        test_invalid_requirements,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(
        "\nPHASE 11.4F-6 INTEGRATED COVERAGE "
        "REPORTER TESTS PASSED"
    )
    print(f"Tests passed: {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
