
"""Phase 11.4F-2 question coverage evaluator tests."""

from dataclasses import replace

from src.recovery.question_coverage_evaluator import (
    CoverageDecision,
    QuestionCoverageEvaluator,
)
from src.recovery.question_requirements import (
    CoverageStatus,
    QuestionCoverageAssessment,
    QuestionRequirements,
    RelationshipCoverage,
    RequiredRelationship,
)


EVALUATOR = QuestionCoverageEvaluator()


def relation(rid, subject, predicate, obj):
    return RequiredRelationship(
        relationship_id=rid,
        subject=subject,
        predicate=predicate,
        object=obj,
    )


def coverage(rid, status, citations=(), bindings=()):
    return RelationshipCoverage(
        relationship_id=rid,
        status=status,
        supporting_chunk_ids=tuple(citations),
        bindings=tuple(bindings),
    )


def base_assessment():
    requirements = QuestionRequirements(
        question=(
            "Which university did the author "
            "of The Silver Atlas attend?"
        ),
        relationships=(
            relation(
                "r1", "The Silver Atlas",
                "authored_by", "?author",
            ),
            relation(
                "r2", "?author",
                "attended", "?university",
            ),
        ),
        answer_variable="?university",
    )

    return QuestionCoverageAssessment(
        requirements=requirements,
        coverage=(
            coverage(
                "r1", CoverageStatus.SUPPORTED,
                ("book",),
                (("?author", "Elena Marsh"),),
            ),
            coverage(
                "r2", CoverageStatus.SUPPORTED,
                ("education",),
                (
                    ("?author", "Elena Marsh"),
                    ("?university", "Northbridge University"),
                ),
            ),
        ),
        assessed_chunk_ids=(
            "book", "education", "negative",
        ),
        explanation="Synthetic assessment.",
    )


def test_complete_chain():
    result = EVALUATOR.evaluate(base_assessment())

    assert result.is_structurally_complete
    assert result.answer_value == "Northbridge University"
    assert result.supported_relationship_ids == ("r1", "r2")


def test_missing_second_hop():
    original = base_assessment()

    modified = replace(
        original,
        coverage=(
            original.coverage[0],
            coverage("r2", CoverageStatus.MISSING),
        ),
    )

    result = EVALUATOR.evaluate(modified)

    assert result.decision == CoverageDecision.INCOMPLETE
    assert result.missing_relationship_ids == ("r2",)
    assert "?university" in result.unbound_variables


def test_missing_first_hop():
    original = base_assessment()

    modified = replace(
        original,
        coverage=(
            coverage("r1", CoverageStatus.MISSING),
            original.coverage[1],
        ),
    )

    result = EVALUATOR.evaluate(modified)

    assert result.decision == CoverageDecision.INCOMPLETE
    assert result.missing_relationship_ids == ("r1",)


def test_disconnected_entities():
    original = base_assessment()

    modified = replace(
        original,
        coverage=(
            original.coverage[0],
            coverage(
                "r2", CoverageStatus.SUPPORTED,
                ("education",),
                (
                    ("?author", "Daniel Reed"),
                    ("?university", "Northbridge University"),
                ),
            ),
        ),
    )

    result = EVALUATOR.evaluate(modified)

    assert result.decision == CoverageDecision.CONFLICTING
    assert result.inconsistent_variables == ("?author",)


def test_explicit_conflict():
    original = base_assessment()

    modified = replace(
        original,
        coverage=(
            original.coverage[0],
            coverage(
                "r2", CoverageStatus.CONFLICTING,
                ("education", "negative"),
            ),
        ),
    )

    result = EVALUATOR.evaluate(modified)

    assert result.decision == CoverageDecision.CONFLICTING
    assert result.conflicting_relationship_ids == ("r2",)


def test_unknown_relationship():
    original = base_assessment()

    modified = replace(
        original,
        coverage=(
            original.coverage[0],
            coverage("r2", CoverageStatus.UNKNOWN),
        ),
    )

    result = EVALUATOR.evaluate(modified)

    assert result.decision == CoverageDecision.UNKNOWN
    assert result.unknown_relationship_ids == ("r2",)


def test_unbound_answer():
    original = base_assessment()

    modified = replace(
        original,
        coverage=(
            original.coverage[0],
            coverage(
                "r2", CoverageStatus.SUPPORTED,
                ("education",),
                (("?author", "Elena Marsh"),),
            ),
        ),
    )

    result = EVALUATOR.evaluate(modified)

    assert result.decision == CoverageDecision.UNKNOWN
    assert result.answer_value is None
    assert "?university" in result.unbound_variables


def test_single_hop():
    req = QuestionRequirements(
        question="Where did Elena Marsh study?",
        relationships=(
            relation(
                "r1", "Elena Marsh",
                "attended", "?university",
            ),
        ),
        answer_variable="?university",
    )

    assessed = QuestionCoverageAssessment(
        requirements=req,
        coverage=(
            coverage(
                "r1", CoverageStatus.SUPPORTED,
                ("education",),
                (("?university", "Northbridge University"),),
            ),
        ),
        assessed_chunk_ids=("education",),
        explanation="Single-hop example.",
    )

    result = EVALUATOR.evaluate(assessed)

    assert result.is_structurally_complete


def test_three_hop():
    req = QuestionRequirements(
        question="Which city is the author's company in?",
        relationships=(
            relation("r1", "Book A", "author", "?person"),
            relation("r2", "?person", "works_at", "?company"),
            relation("r3", "?company", "located_in", "?city"),
        ),
        answer_variable="?city",
    )

    assessed = QuestionCoverageAssessment(
        requirements=req,
        coverage=(
            coverage(
                "r1", CoverageStatus.SUPPORTED,
                ("c1",),
                (("?person", "Person A"),),
            ),
            coverage(
                "r2", CoverageStatus.SUPPORTED,
                ("c2",),
                (
                    ("?person", "Person A"),
                    ("?company", "Company B"),
                ),
            ),
            coverage(
                "r3", CoverageStatus.SUPPORTED,
                ("c3",),
                (
                    ("?company", "Company B"),
                    ("?city", "City C"),
                ),
            ),
        ),
        assessed_chunk_ids=("c1", "c2", "c3"),
        explanation="Three-hop example.",
    )

    result = EVALUATOR.evaluate(assessed)

    assert result.is_structurally_complete
    assert result.answer_value == "City C"


def test_invalid_input():
    try:
        EVALUATOR.evaluate("invalid")
    except TypeError:
        return

    raise AssertionError("Expected TypeError.")


def main():
    tests = [
        test_complete_chain,
        test_missing_second_hop,
        test_missing_first_hop,
        test_disconnected_entities,
        test_explicit_conflict,
        test_unknown_relationship,
        test_unbound_answer,
        test_single_hop,
        test_three_hop,
        test_invalid_input,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(
        "\nPHASE 11.4F-2 QUESTION COVERAGE "
        "EVALUATOR TESTS PASSED"
    )
    print(f"Tests passed: {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
