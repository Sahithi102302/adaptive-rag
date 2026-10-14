
"""Phase 11.4F-4 relational evidence join tests."""

from src.recovery.relational_evidence_joiner import (
    JoinDecision,
    RelationalEvidenceJoiner,
)
from src.recovery.question_requirements import (
    QuestionRequirements,
    RequiredRelationship,
)


JOINER = RelationalEvidenceJoiner()


def rel(rid, subject, predicate, obj):
    return RequiredRelationship(
        relationship_id=rid,
        subject=subject,
        predicate=predicate,
        object=obj,
    )


def req(*relations, answer="?university"):
    return QuestionRequirements(
        question="Synthetic relational question",
        relationships=tuple(relations),
        answer_variable=answer,
    )


def two_hop():
    return req(
        rel("r1", "The Silver Atlas", "authored_by", "?author"),
        rel("r2", "?author", "attended", "?university"),
    )


def test_single_hop():
    requirements = req(
        rel("r1", "Elena Marsh", "attended", "?university")
    )
    report = JOINER.join(
        requirements,
        {"e": "Elena Marsh attended Northbridge University."},
    )
    assert report.decision == JoinDecision.UNIQUE_ANSWER
    assert report.answer_values == ("Northbridge University",)


def test_complete_two_hop():
    report = JOINER.join(
        two_hop(),
        {
            "book": "The Silver Atlas was written by Elena Marsh.",
            "education": "Elena Marsh attended Northbridge University.",
        },
    )
    assert report.decision == JoinDecision.UNIQUE_ANSWER
    assert len(report.paths) == 1
    assert report.paths[0].binding("?author") == "Elena Marsh"
    assert report.paths[0].supporting_chunk_ids == (
        "book", "education"
    )


def test_irrelevant_distractor():
    report = JOINER.join(
        two_hop(),
        {
            "book": "The Silver Atlas was written by Elena Marsh.",
            "a": "Elena Marsh attended Northbridge University.",
            "b": "Daniel Reed attended Lakeside College.",
        },
    )
    assert report.decision == JoinDecision.UNIQUE_ANSWER
    assert report.answer_values == ("Northbridge University",)
    assert len(report.paths) == 1


def test_disconnected_entities():
    report = JOINER.join(
        two_hop(),
        {
            "book": "The Silver Atlas was written by Elena Marsh.",
            "education": "Daniel Reed attended Lakeside College.",
        },
    )
    assert report.decision == JoinDecision.NO_COMPLETE_PATH
    assert report.paths == ()


def test_missing_second_hop():
    report = JOINER.join(
        two_hop(),
        {"book": "The Silver Atlas was written by Elena Marsh."},
    )
    assert report.decision == JoinDecision.NO_COMPLETE_PATH
    assert report.unmatched_relationship_ids == ("r2",)


def test_missing_first_hop():
    report = JOINER.join(
        two_hop(),
        {"education": "Elena Marsh attended Northbridge University."},
    )
    assert report.decision == JoinDecision.NO_COMPLETE_PATH
    assert report.unmatched_relationship_ids == ("r1",)


def test_multiple_answers():
    requirements = req(
        rel("r1", "Elena Marsh", "attended", "?university")
    )
    report = JOINER.join(
        requirements,
        {
            "a": "Elena Marsh attended Northbridge University.",
            "b": "Elena Marsh attended Lakeside College.",
        },
    )
    assert report.decision == JoinDecision.AMBIGUOUS
    assert len(report.answer_values) == 2


def test_duplicate_fact_same_answer():
    requirements = req(
        rel("r1", "Elena Marsh", "attended", "?university")
    )
    report = JOINER.join(
        requirements,
        {
            "a": "Elena Marsh attended Northbridge University.",
            "b": "Elena Marsh attended Northbridge University.",
        },
    )
    assert report.decision == JoinDecision.UNIQUE_ANSWER
    assert len(report.paths) == 2
    assert len(report.answer_values) == 1


def test_three_hop_unsupported_predicate():
    requirements = req(
        rel("r1", "The Silver Atlas", "authored_by", "?author"),
        rel("r2", "?author", "attended", "?university"),
        rel("r3", "?university", "located_in", "?city"),
        answer="?city",
    )
    report = JOINER.join(
        requirements,
        {
            "book": "The Silver Atlas was written by Elena Marsh.",
            "education": "Elena Marsh attended Northbridge University.",
            "city": "Northbridge University is located in Boston.",
        },
    )
    assert report.decision == JoinDecision.UNKNOWN
    assert report.unmatched_relationship_ids == ("r3",)


def test_three_relationship_join():
    requirements = req(
        rel("r1", "The Silver Atlas", "authored_by", "?author"),
        rel("r2", "?author", "attended", "?university"),
        rel("r3", "?author", "attended", "?other_school"),
        answer="?university",
    )
    report = JOINER.join(
        requirements,
        {
            "book": "The Silver Atlas was written by Elena Marsh.",
            "education": "Elena Marsh attended Northbridge University.",
        },
    )
    assert report.decision == JoinDecision.UNIQUE_ANSWER
    assert len(report.paths) == 1


def test_repeated_variable_must_agree():
    requirements = req(
        rel("r1", "?person", "attended", "?person"),
        answer="?person",
    )
    report = JOINER.join(
        requirements,
        {"a": "Elena Marsh attended Northbridge University."},
    )
    assert report.decision == JoinDecision.NO_COMPLETE_PATH


def test_case_insensitive_join():
    report = JOINER.join(
        two_hop(),
        {
            "book": "The Silver Atlas was written by ELENA MARSH.",
            "education": "Elena Marsh attended Northbridge University.",
        },
    )
    assert report.decision == JoinDecision.UNIQUE_ANSWER


def test_negated_fact_not_positive_support():
    requirements = req(
        rel("r1", "Elena Marsh", "attended", "?university")
    )
    report = JOINER.join(
        requirements,
        {
            "negative": (
                "Elena Marsh never attended Northbridge University."
            )
        },
    )
    assert report.decision == JoinDecision.NO_COMPLETE_PATH


def test_unexamined_contradiction_limitation():
    requirements = req(
        rel("r1", "Elena Marsh", "attended", "?university")
    )
    report = JOINER.join(
        requirements,
        {
            "positive": "Elena Marsh attended Northbridge University.",
            "negative": (
                "Elena Marsh never attended Northbridge University."
            ),
        },
    )
    # Known limitation: the positive-path join alone cannot
    # detect the contradiction.
    assert report.decision == JoinDecision.UNIQUE_ANSWER


def test_invalid_requirements():
    try:
        JOINER.join("invalid", {})
    except TypeError:
        return
    raise AssertionError("Expected TypeError.")


def main():
    tests = [
        test_single_hop,
        test_complete_two_hop,
        test_irrelevant_distractor,
        test_disconnected_entities,
        test_missing_second_hop,
        test_missing_first_hop,
        test_multiple_answers,
        test_duplicate_fact_same_answer,
        test_three_hop_unsupported_predicate,
        test_three_relationship_join,
        test_repeated_variable_must_agree,
        test_case_insensitive_join,
        test_negated_fact_not_positive_support,
        test_unexamined_contradiction_limitation,
        test_invalid_requirements,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(
        "\nPHASE 11.4F-4 RELATIONAL EVIDENCE "
        "JOINER TESTS PASSED"
    )
    print(f"Tests passed: {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
