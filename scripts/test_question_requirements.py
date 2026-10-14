
"""Phase 11.4F-1 question requirements contract tests."""

from dataclasses import replace

from src.recovery.question_requirements import (
    CoverageStatus,
    QuestionCoverageAssessment,
    QuestionRequirements,
    RelationshipCoverage,
    RequiredRelationship,
)


def relation(rid, subject, predicate, obj):
    return RequiredRelationship(
        relationship_id=rid,
        subject=subject,
        predicate=predicate,
        object=obj,
    )


def covered(rid, status, citations=(), bindings=()):
    return RelationshipCoverage(
        relationship_id=rid,
        status=status,
        supporting_chunk_ids=tuple(citations),
        bindings=tuple(bindings),
    )


def requirements(*relationships, answer="?university"):
    return QuestionRequirements(
        question="Which university did the author attend?",
        relationships=tuple(relationships),
        answer_variable=answer,
    )


def two_hop_requirements():
    return requirements(
        relation(
            "r1", "The Silver Atlas",
            "authored_by", "?author",
        ),
        relation(
            "r2", "?author",
            "attended", "?university",
        ),
    )


def assessment(req, items, ids):
    return QuestionCoverageAssessment(
        requirements=req,
        coverage=tuple(items),
        assessed_chunk_ids=tuple(ids),
        explanation="Synthetic coverage assessment.",
    )


def expect_error(error_type, operation):
    try:
        operation()
    except error_type:
        return

    raise AssertionError(
        f"Expected {error_type.__name__}."
    )


def complete_two_hop():
    return assessment(
        two_hop_requirements(),
        (
            covered(
                "r1", CoverageStatus.SUPPORTED,
                ("book",),
                (("?author", "Elena Marsh"),),
            ),
            covered(
                "r2", CoverageStatus.SUPPORTED,
                ("education",),
                (
                    ("?author", "Elena Marsh"),
                    ("?university", "Northbridge University"),
                ),
            ),
        ),
        ("book", "education"),
    )


def test_single_hop():
    req = requirements(
        relation(
            "r1", "Elena Marsh",
            "attended", "?university",
        ),
    )

    result = assessment(
        req,
        (
            covered(
                "r1", CoverageStatus.SUPPORTED,
                ("education",),
                (("?university", "Northbridge University"),),
            ),
        ),
        ("education",),
    )

    assert result.structurally_complete


def test_complete_two_hop():
    result = complete_two_hop()
    assert result.structurally_complete
    assert result.bindings_consistent


def test_three_hop():
    req = requirements(
        relation("r1", "Book A", "authored_by", "?author"),
        relation("r2", "?author", "worked_at", "?company"),
        relation("r3", "?company", "located_in", "?city"),
        answer="?city",
    )

    result = assessment(
        req,
        (
            covered(
                "r1", CoverageStatus.SUPPORTED,
                ("c1",),
                (("?author", "Person A"),),
            ),
            covered(
                "r2", CoverageStatus.SUPPORTED,
                ("c2",),
                (
                    ("?author", "Person A"),
                    ("?company", "Company B"),
                ),
            ),
            covered(
                "r3", CoverageStatus.SUPPORTED,
                ("c3",),
                (
                    ("?company", "Company B"),
                    ("?city", "City C"),
                ),
            ),
        ),
        ("c1", "c2", "c3"),
    )

    assert result.structurally_complete


def test_missing_second_hop():
    original = complete_two_hop()

    result = replace(
        original,
        coverage=(
            original.coverage[0],
            covered("r2", CoverageStatus.MISSING),
        ),
    )

    assert result.has_missing_links
    assert not result.structurally_complete


def test_missing_first_hop():
    original = complete_two_hop()

    result = replace(
        original,
        coverage=(
            covered("r1", CoverageStatus.MISSING),
            original.coverage[1],
        ),
    )

    assert not result.structurally_complete


def test_disconnected_entities():
    original = complete_two_hop()

    result = replace(
        original,
        coverage=(
            original.coverage[0],
            covered(
                "r2", CoverageStatus.SUPPORTED,
                ("education",),
                (
                    ("?author", "Daniel Reed"),
                    ("?university", "Northbridge University"),
                ),
            ),
        ),
    )

    assert not result.bindings_consistent
    assert not result.structurally_complete


def test_conflicting_relationship():
    original = complete_two_hop()

    result = replace(
        original,
        coverage=(
            original.coverage[0],
            covered(
                "r2", CoverageStatus.CONFLICTING,
                ("education", "negative"),
            ),
        ),
        assessed_chunk_ids=(
            "book", "education", "negative",
        ),
    )

    assert result.has_conflicts
    assert not result.structurally_complete


def test_unknown_relationship():
    original = complete_two_hop()

    result = replace(
        original,
        coverage=(
            original.coverage[0],
            covered("r2", CoverageStatus.UNKNOWN),
        ),
    )

    assert not result.structurally_complete


def test_missing_answer_binding():
    original = complete_two_hop()

    result = replace(
        original,
        coverage=(
            original.coverage[0],
            covered(
                "r2", CoverageStatus.SUPPORTED,
                ("education",),
                (("?author", "Elena Marsh"),),
            ),
        ),
    )

    assert not result.structurally_complete


def test_duplicate_relationship_ids():
    expect_error(
        ValueError,
        lambda: requirements(
            relation("r1", "Book A", "author", "?author"),
            relation("r1", "?author", "attended", "?university"),
        ),
    )


def test_missing_coverage_rejected():
    original = complete_two_hop()

    expect_error(
        ValueError,
        lambda: replace(
            original,
            coverage=(original.coverage[0],),
        ),
    )


def test_unknown_citation_rejected():
    original = complete_two_hop()

    expect_error(
        ValueError,
        lambda: replace(
            original,
            coverage=(
                original.coverage[0],
                covered(
                    "r2", CoverageStatus.SUPPORTED,
                    ("invented",),
                    (
                        ("?author", "Elena Marsh"),
                        ("?university", "Northbridge University"),
                    ),
                ),
            ),
        ),
    )


def test_invalid_binding_variable():
    expect_error(
        ValueError,
        lambda: covered(
            "r1", CoverageStatus.SUPPORTED,
            ("book",),
            (("author", "Elena Marsh"),),
        ),
    )


def test_unrelated_binding_rejected():
    original = complete_two_hop()

    expect_error(
        ValueError,
        lambda: replace(
            original,
            coverage=(
                covered(
                    "r1", CoverageStatus.SUPPORTED,
                    ("book",),
                    (("?university", "Northbridge University"),),
                ),
                original.coverage[1],
            ),
        ),
    )


def test_supported_requires_citation():
    expect_error(
        ValueError,
        lambda: covered(
            "r1", CoverageStatus.SUPPORTED,
        ),
    )


def test_conflict_requires_two_citations():
    expect_error(
        ValueError,
        lambda: covered(
            "r1", CoverageStatus.CONFLICTING,
            ("education",),
        ),
    )


def test_missing_cannot_claim_bindings():
    expect_error(
        ValueError,
        lambda: covered(
            "r1", CoverageStatus.MISSING,
            (),
            (("?author", "Elena Marsh"),),
        ),
    )


def main():
    tests = [
        test_single_hop,
        test_complete_two_hop,
        test_three_hop,
        test_missing_second_hop,
        test_missing_first_hop,
        test_disconnected_entities,
        test_conflicting_relationship,
        test_unknown_relationship,
        test_missing_answer_binding,
        test_duplicate_relationship_ids,
        test_missing_coverage_rejected,
        test_unknown_citation_rejected,
        test_invalid_binding_variable,
        test_unrelated_binding_rejected,
        test_supported_requires_citation,
        test_conflict_requires_two_citations,
        test_missing_cannot_claim_bindings,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(
        "\nPHASE 11.4F-1 QUESTION REQUIREMENTS "
        "CONTRACT TESTS PASSED"
    )
    print(f"Tests passed: {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
