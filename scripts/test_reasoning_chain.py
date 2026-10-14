
"""Phase 11.4D-1 reasoning-chain contract tests."""

from dataclasses import replace

from src.recovery.reasoning_chain import (
    EvidenceContradiction,
    ReasoningChainAssessment,
    ReasoningStep,
    ReasoningStepStatus,
)
from src.recovery.semantic_verification import SemanticSupport


def step(
    step_id,
    status,
    citations,
    relationship="Test relationship",
):
    return ReasoningStep(
        step_id=step_id,
        relationship=relationship,
        status=status,
        supporting_chunk_ids=tuple(citations),
    )


def valid_assessment():
    return ReasoningChainAssessment(
        candidate_answer="Northbridge University",
        steps=(
            step(
                1,
                ReasoningStepStatus.SUPPORTED,
                ("book",),
                "The Silver Atlas was written by Elena Marsh.",
            ),
            step(
                2,
                ReasoningStepStatus.SUPPORTED,
                ("education",),
                "Elena Marsh attended Northbridge University.",
            ),
        ),
        contradictions=(),
        support=SemanticSupport.SUPPORTED,
        assessed_chunk_ids=("book", "education"),
        explanation="Both relationships are supported.",
        assessor_name="test_assessor",
    )


def expect_error(error_type, operation):
    try:
        operation()
    except error_type:
        return
    raise AssertionError(
        f"Expected {error_type.__name__}."
    )


def test_complete_chain():
    result = valid_assessment()

    assert result.support == SemanticSupport.SUPPORTED
    assert len(result.steps) == 2
    assert not result.has_missing_steps
    assert not result.has_conflicts


def test_missing_second_hop():
    result = replace(
        valid_assessment(),
        candidate_answer=None,
        steps=(
            step(
                1,
                ReasoningStepStatus.SUPPORTED,
                ("book",),
            ),
            step(
                2,
                ReasoningStepStatus.MISSING,
                (),
            ),
        ),
        support=SemanticSupport.PARTIAL,
    )

    assert result.has_missing_steps
    assert result.support == SemanticSupport.PARTIAL


def test_missing_step_cannot_be_supported():
    expect_error(
        ValueError,
        lambda: replace(
            valid_assessment(),
            steps=(
                step(
                    1,
                    ReasoningStepStatus.MISSING,
                    (),
                ),
            ),
        ),
    )


def test_supported_requires_answer():
    expect_error(
        ValueError,
        lambda: replace(
            valid_assessment(),
            candidate_answer=None,
        ),
    )


def test_supported_requires_steps():
    expect_error(
        ValueError,
        lambda: replace(
            valid_assessment(),
            steps=(),
        ),
    )


def test_supported_step_requires_citation():
    expect_error(
        ValueError,
        lambda: step(
            1,
            ReasoningStepStatus.SUPPORTED,
            (),
        ),
    )


def test_missing_step_rejects_citation():
    expect_error(
        ValueError,
        lambda: step(
            1,
            ReasoningStepStatus.MISSING,
            ("book",),
        ),
    )


def test_unknown_citation_rejected():
    expect_error(
        ValueError,
        lambda: replace(
            valid_assessment(),
            steps=(
                step(
                    1,
                    ReasoningStepStatus.SUPPORTED,
                    ("invented",),
                ),
            ),
        ),
    )


def test_duplicate_step_ids_rejected():
    expect_error(
        ValueError,
        lambda: replace(
            valid_assessment(),
            steps=(
                step(
                    1,
                    ReasoningStepStatus.SUPPORTED,
                    ("book",),
                ),
                step(
                    1,
                    ReasoningStepStatus.SUPPORTED,
                    ("education",),
                ),
            ),
        ),
    )


def test_duplicate_chunk_ids_rejected():
    expect_error(
        ValueError,
        lambda: replace(
            valid_assessment(),
            assessed_chunk_ids=("book", "book"),
        ),
    )


def test_contradiction_blocks_supported():
    contradiction = EvidenceContradiction(
        explanation="Conflicting university claims.",
        supporting_chunk_ids=(
            "education",
            "education_negative",
        ),
    )

    expect_error(
        ValueError,
        lambda: replace(
            valid_assessment(),
            contradictions=(contradiction,),
            assessed_chunk_ids=(
                "book",
                "education",
                "education_negative",
            ),
        ),
    )


def test_conflicting_step_requires_two_citations():
    expect_error(
        ValueError,
        lambda: step(
            1,
            ReasoningStepStatus.CONFLICTING,
            ("education",),
        ),
    )


def test_valid_conflict_assessment():
    contradiction = EvidenceContradiction(
        explanation="The two passages disagree.",
        supporting_chunk_ids=("positive", "negative"),
    )

    result = ReasoningChainAssessment(
        candidate_answer=None,
        steps=(
            step(
                1,
                ReasoningStepStatus.CONFLICTING,
                ("positive", "negative"),
            ),
        ),
        contradictions=(contradiction,),
        support=SemanticSupport.UNKNOWN,
        assessed_chunk_ids=("positive", "negative"),
        explanation="Conflicting evidence.",
        assessor_name="test_assessor",
    )

    assert result.has_conflicts
    assert result.support == SemanticSupport.UNKNOWN


def test_conflicting_step_requires_record():
    expect_error(
        ValueError,
        lambda: ReasoningChainAssessment(
            candidate_answer=None,
            steps=(
                step(
                    1,
                    ReasoningStepStatus.CONFLICTING,
                    ("positive", "negative"),
                ),
            ),
            contradictions=(),
            support=SemanticSupport.UNKNOWN,
            assessed_chunk_ids=("positive", "negative"),
            explanation="Conflict detected.",
            assessor_name="test_assessor",
        ),
    )


def test_unknown_contradiction_citation_rejected():
    contradiction = EvidenceContradiction(
        explanation="Two conflicting claims.",
        supporting_chunk_ids=("education", "invented"),
    )

    expect_error(
        ValueError,
        lambda: replace(
            valid_assessment(),
            candidate_answer=None,
            support=SemanticSupport.UNKNOWN,
            contradictions=(contradiction,),
        ),
    )


def main():
    tests = [
        test_complete_chain,
        test_missing_second_hop,
        test_missing_step_cannot_be_supported,
        test_supported_requires_answer,
        test_supported_requires_steps,
        test_supported_step_requires_citation,
        test_missing_step_rejects_citation,
        test_unknown_citation_rejected,
        test_duplicate_step_ids_rejected,
        test_duplicate_chunk_ids_rejected,
        test_contradiction_blocks_supported,
        test_conflicting_step_requires_two_citations,
        test_valid_conflict_assessment,
        test_conflicting_step_requires_record,
        test_unknown_contradiction_citation_rejected,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(
        "\nPHASE 11.4D-1 REASONING-CHAIN "
        "CONTRACT TESTS PASSED"
    )
    print(f"Tests passed: {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
