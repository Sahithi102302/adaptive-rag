
"""Phase 11.4A semantic verification contract tests."""

from dataclasses import replace

from src.recovery.semantic_verification import (
    SemanticEvidenceAssessment,
    SemanticFinding,
    SemanticFindingType,
    SemanticSupport,
)


def expect_value_error(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("Expected ValueError.")


def make_support_finding():
    return SemanticFinding(
        finding_type=SemanticFindingType.SUPPORT,
        explanation="The cited evidence supports the stated relationship.",
        supporting_chunk_ids=("chunk_a",),
    )


def make_assessment():
    return SemanticEvidenceAssessment(
        support=SemanticSupport.SUPPORTED,
        findings=(make_support_finding(),),
        assessed_chunk_ids=("chunk_a", "chunk_b"),
        explanation="The candidate appears to support the question.",
        assessor_name="test_assessor",
    )


def test_valid_supported_assessment():
    result = make_assessment()
    assert result.support == SemanticSupport.SUPPORTED
    assert not result.has_contradiction
    assert not result.has_missing_link


def test_unknown_chunk_reference_rejected():
    finding = replace(
        make_support_finding(),
        supporting_chunk_ids=("missing_chunk",),
    )
    expect_value_error(
        lambda: replace(make_assessment(), findings=(finding,))
    )


def test_supported_without_citation_rejected():
    expect_value_error(
        lambda: SemanticEvidenceAssessment(
            support=SemanticSupport.SUPPORTED,
            findings=(),
            assessed_chunk_ids=("chunk_a",),
            explanation="Unsupported claim of full support.",
            assessor_name="test_assessor",
        )
    )


def test_contradiction_detection():
    contradiction = SemanticFinding(
        finding_type=SemanticFindingType.CONTRADICTION,
        explanation="Two cited passages make conflicting claims.",
        supporting_chunk_ids=("chunk_a", "chunk_b"),
    )
    result = replace(
        make_assessment(),
        findings=(make_support_finding(), contradiction),
    )
    assert result.has_contradiction


def test_missing_link_detection():
    missing = SemanticFinding(
        finding_type=SemanticFindingType.MISSING_LINK,
        explanation="The connection between the entities is not established.",
        supporting_chunk_ids=(),
    )
    result = SemanticEvidenceAssessment(
        support=SemanticSupport.PARTIAL,
        findings=(missing,),
        assessed_chunk_ids=("chunk_a",),
        explanation="One reasoning link appears missing.",
        assessor_name="test_assessor",
    )
    assert result.has_missing_link


def test_duplicate_chunk_rejected():
    expect_value_error(
        lambda: replace(
            make_assessment(),
            assessed_chunk_ids=("chunk_a", "chunk_a"),
        )
    )


def main():
    tests = [
        test_valid_supported_assessment,
        test_unknown_chunk_reference_rejected,
        test_supported_without_citation_rejected,
        test_contradiction_detection,
        test_missing_link_detection,
        test_duplicate_chunk_rejected,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("\nPHASE 11.4A SEMANTIC VERIFICATION CONTRACT TESTS PASSED")


if __name__ == "__main__":
    main()
