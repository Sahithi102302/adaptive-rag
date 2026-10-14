
"""Phase 11.4F-3 deterministic relationship matcher tests."""

from src.recovery.evidence_relationship_matcher import (
    EvidenceRelationshipMatcher,
)
from src.recovery.question_coverage_evaluator import (
    CoverageDecision,
    QuestionCoverageEvaluator,
)
from src.recovery.question_requirements import (
    CoverageStatus,
    QuestionRequirements,
    RequiredRelationship,
)


MATCHER = EvidenceRelationshipMatcher()
EVALUATOR = QuestionCoverageEvaluator()


def req(*relations, answer="?university"):
    return QuestionRequirements(
        question="Which university did the author attend?",
        relationships=tuple(relations),
        answer_variable=answer,
    )


def rel(rid, subject, predicate, obj):
    return RequiredRelationship(
        relationship_id=rid,
        subject=subject,
        predicate=predicate,
        object=obj,
    )


def two_hop():
    return req(
        rel("r1", "The Silver Atlas", "authored_by", "?author"),
        rel("r2", "?author", "attended", "?university"),
    )


def test_complete_two_hop():
    evidence = {
        "book": "The Silver Atlas was written by Elena Marsh.",
        "education": "Elena Marsh attended Northbridge University.",
    }

    result = EVALUATOR.evaluate(
        MATCHER.assess(two_hop(), evidence)
    )

    assert result.decision == CoverageDecision.STRUCTURALLY_COMPLETE
    assert result.answer_value == "Northbridge University"


def test_missing_second_hop():
    assessment = MATCHER.assess(
        two_hop(),
        {"book": "The Silver Atlas was written by Elena Marsh."},
    )

    assert assessment.coverage[1].status == CoverageStatus.MISSING


def test_missing_first_hop():
    assessment = MATCHER.assess(
        two_hop(),
        {"education": "Elena Marsh attended Northbridge University."},
    )

    assert assessment.coverage[0].status == CoverageStatus.MISSING


def test_disconnected_author():
    evidence = {
        "book": "The Silver Atlas was written by Elena Marsh.",
        "education": "Daniel Reed attended Northbridge University.",
    }

    report = EVALUATOR.evaluate(
        MATCHER.assess(two_hop(), evidence)
    )

    assert report.decision == CoverageDecision.CONFLICTING
    assert report.inconsistent_variables == ("?author",)


def test_explicit_contradiction():
    evidence = {
        "book": "The Silver Atlas was written by Elena Marsh.",
        "positive": "Elena Marsh attended Northbridge University.",
        "negative": "Elena Marsh never attended Northbridge University.",
    }

    assessment = MATCHER.assess(two_hop(), evidence)

    assert assessment.coverage[1].status == CoverageStatus.CONFLICTING
    assert set(assessment.coverage[1].supporting_chunk_ids) == {
        "positive", "negative"
    }



def test_invented_university_not_supported():
    # The evidence identifies the author but provides
    # no evidence of which university the author attended.
    # The matcher must not invent a university.

    requirements = two_hop()

    assessment = MATCHER.assess(
        requirements,
        {
            "book": (
                "The Silver Atlas was written by Elena Marsh."
            )
        },
    )

    assert (
        assessment.coverage[0].status
        == CoverageStatus.SUPPORTED
    )

    assert (
        assessment.coverage[1].status
        == CoverageStatus.MISSING
    )



def test_unknown_predicate():
    requirements = req(
        rel(
            "r1", "Elena Marsh",
            "graduated_from", "?university",
        ),
    )

    assessment = MATCHER.assess(
        requirements,
        {"education": "Elena Marsh graduated from Northbridge University."},
    )

    assert assessment.coverage[0].status == CoverageStatus.UNKNOWN


def test_multiple_possible_universities_unknown():
    requirements = req(
        rel("r1", "Elena Marsh", "attended", "?university"),
    )

    assessment = MATCHER.assess(
        requirements,
        {
            "a": "Elena Marsh attended Northbridge University.",
            "b": "Elena Marsh attended Lakeside College.",
        },
    )

    assert assessment.coverage[0].status == CoverageStatus.UNKNOWN


def test_single_hop():
    requirements = req(
        rel("r1", "Elena Marsh", "attended", "?university"),
    )

    report = EVALUATOR.evaluate(
        MATCHER.assess(
            requirements,
            {"education": "Elena Marsh attended Northbridge University."},
        )
    )

    assert report.is_structurally_complete


def test_case_normalization():
    requirements = req(
        rel("r1", "elena marsh", "attended", "?university"),
    )

    assessment = MATCHER.assess(
        requirements,
        {"education": "Elena Marsh attended Northbridge University."},
    )

    assert assessment.coverage[0].status == CoverageStatus.SUPPORTED


def test_unsupported_paraphrase():
    requirements = req(
        rel("r1", "Elena Marsh", "attended", "?university"),
    )

    assessment = MATCHER.assess(
        requirements,
        {"education": "Elena Marsh studied at Northbridge University."},
    )

    assert assessment.coverage[0].status == CoverageStatus.MISSING


def test_three_hop():
    # Third predicate is outside the supported extraction grammar.
    requirements = req(
        rel("r1", "The Silver Atlas", "authored_by", "?author"),
        rel("r2", "?author", "attended", "?university"),
        rel("r3", "?university", "located_in", "?city"),
        answer="?city",
    )

    assessment = MATCHER.assess(
        requirements,
        {
            "book": "The Silver Atlas was written by Elena Marsh.",
            "education": "Elena Marsh attended Northbridge University.",
            "location": "Northbridge University is located in Boston.",
        },
    )

    assert assessment.coverage[2].status == CoverageStatus.UNKNOWN


def main():
    tests = [
        test_complete_two_hop,
        test_missing_second_hop,
        test_missing_first_hop,
        test_disconnected_author,
        test_explicit_contradiction,
        test_invented_university_not_supported,
        test_unknown_predicate,
        test_multiple_possible_universities_unknown,
        test_single_hop,
        test_case_normalization,
        test_unsupported_paraphrase,
        test_three_hop,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(
        "\nPHASE 11.4F-3 EVIDENCE RELATIONSHIP "
        "MATCHER TESTS PASSED"
    )
    print(f"Tests passed: {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
