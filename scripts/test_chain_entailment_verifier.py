
"""Phase 11.4E-3 chain-level verification tests."""

from src.recovery.chain_entailment_verifier import (
    ChainEntailmentVerifier,
)
from src.recovery.evidence_entailment import EntailmentLabel
from src.recovery.reasoning_chain import (
    ReasoningChainAssessment,
    ReasoningStep,
    ReasoningStepStatus,
)
from src.recovery.semantic_verification import SemanticSupport


VERIFIER = ChainEntailmentVerifier()

QUESTION_EVIDENCE = {
    "book": "The Silver Atlas was written by Elena Marsh.",
    "education": "Elena Marsh attended Northbridge University.",
}


def step(number, claim, citations):
    return ReasoningStep(
        step_id=number,
        relationship=claim,
        status=ReasoningStepStatus.SUPPORTED,
        supporting_chunk_ids=tuple(citations),
    )


def chain(steps, evidence, answer="Northbridge University"):
    return ReasoningChainAssessment(
        candidate_answer=answer,
        steps=tuple(steps),
        contradictions=(),
        support=SemanticSupport.SUPPORTED,
        assessed_chunk_ids=tuple(evidence),
        explanation="Synthetic proposed reasoning chain.",
        assessor_name="synthetic_qwen",
    )


def test_complete_explicit_chain_passes():
    evidence = QUESTION_EVIDENCE

    proposed = chain(
        [
            step(
                1,
                "The Silver Atlas was written by Elena Marsh.",
                ("book",),
            ),
            step(
                2,
                "Elena Marsh attended Northbridge University.",
                ("education",),
            ),
        ],
        evidence,
    )

    result = VERIFIER.verify(proposed, evidence)

    assert result.passes_literal_gate
    assert result.assessment.all_entailed


def test_invented_university_blocked():
    evidence = {"book": QUESTION_EVIDENCE["book"]}

    proposed = chain(
        [
            step(
                1,
                "Elena Marsh attended the University of Cambridge.",
                ("book",),
            )
        ],
        evidence,
        answer="University of Cambridge",
    )

    result = VERIFIER.verify(proposed, evidence)

    assert not result.passes_literal_gate
    assert (
        result.assessment.claim_verifications[0].label
        == EntailmentLabel.NOT_ESTABLISHED
    )


def test_missing_first_hop_claim_blocked():
    evidence = {
        "education": QUESTION_EVIDENCE["education"]
    }

    proposed = chain(
        [
            step(
                1,
                "The Silver Atlas was written by Elena Marsh.",
                ("education",),
            )
        ],
        evidence,
    )

    result = VERIFIER.verify(proposed, evidence)
    assert not result.passes_literal_gate


def test_disconnected_author_claim_blocked():
    evidence = {
        "book": QUESTION_EVIDENCE["book"],
        "education": (
            "Daniel Reed attended Northbridge University."
        ),
    }

    proposed = chain(
        [
            step(
                1,
                "The Silver Atlas was written by Daniel Reed.",
                ("book",),
            ),
            step(
                2,
                "Daniel Reed attended Northbridge University.",
                ("education",),
            ),
        ],
        evidence,
    )

    result = VERIFIER.verify(proposed, evidence)

    assert not result.passes_literal_gate
    assert (
        result.assessment.claim_verifications[0].label
        == EntailmentLabel.NOT_ESTABLISHED
    )


def test_generic_claim_blocked():
    evidence = QUESTION_EVIDENCE

    proposed = chain(
        [
            step(
                1,
                "author attended university",
                ("education",),
            ),
            step(
                2,
                "The Silver Atlas was written by Elena Marsh.",
                ("book",),
            ),
        ],
        evidence,
    )

    result = VERIFIER.verify(proposed, evidence)
    assert not result.passes_literal_gate


def test_missing_step_not_verified_as_entailed():
    evidence = {"book": QUESTION_EVIDENCE["book"]}

    proposed = ReasoningChainAssessment(
        candidate_answer=None,
        steps=(
            step(
                1,
                "The Silver Atlas was written by Elena Marsh.",
                ("book",),
            ),
            ReasoningStep(
                step_id=2,
                relationship=(
                    "Elena Marsh attended a university."
                ),
                status=ReasoningStepStatus.MISSING,
                supporting_chunk_ids=(),
            ),
        ),
        contradictions=(),
        support=SemanticSupport.PARTIAL,
        assessed_chunk_ids=("book",),
        explanation="Missing university evidence.",
        assessor_name="synthetic_qwen",
    )

    result = VERIFIER.verify(proposed, evidence)

    assert not result.passes_literal_gate
    assert result.assessment.has_unverified_claims


def test_mismatched_evidence_ids_rejected():
    proposed = chain(
        [
            step(
                1,
                "The Silver Atlas was written by Elena Marsh.",
                ("book",),
            )
        ],
        {"book": QUESTION_EVIDENCE["book"]},
    )

    try:
        VERIFIER.verify(proposed, QUESTION_EVIDENCE)
    except ValueError:
        return

    raise AssertionError("Expected evidence mismatch rejection.")


def test_omitted_step_limitation():
    # Important: the literal gate cannot detect that a required
    # second hop is absent when the proposer omits it entirely.
    evidence = {"book": QUESTION_EVIDENCE["book"]}

    proposed = chain(
        [
            step(
                1,
                "The Silver Atlas was written by Elena Marsh.",
                ("book",),
            )
        ],
        evidence,
    )

    result = VERIFIER.verify(proposed, evidence)

    assert result.passes_literal_gate
    # This is a known limitation, NOT a semantic success.


def test_unreported_contradiction_limitation():
    # An uncited contradictory passage is not detected by
    # a verifier that checks only proposed cited claims.
    evidence = {
        "book": QUESTION_EVIDENCE["book"],
        "education": QUESTION_EVIDENCE["education"],
        "negative": (
            "Elena Marsh never attended Northbridge University."
        ),
    }

    proposed = chain(
        [
            step(
                1,
                "The Silver Atlas was written by Elena Marsh.",
                ("book",),
            ),
            step(
                2,
                "Elena Marsh attended Northbridge University.",
                ("education",),
            ),
        ],
        evidence,
    )

    result = VERIFIER.verify(proposed, evidence)

    assert result.passes_literal_gate
    # Known limitation: the contradictory passage was omitted.


def main():
    tests = [
        test_complete_explicit_chain_passes,
        test_invented_university_blocked,
        test_missing_first_hop_claim_blocked,
        test_disconnected_author_claim_blocked,
        test_generic_claim_blocked,
        test_missing_step_not_verified_as_entailed,
        test_mismatched_evidence_ids_rejected,
        test_omitted_step_limitation,
        test_unreported_contradiction_limitation,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(
        "\nPHASE 11.4E-3 CHAIN ENTAILMENT "
        "VERIFIER TESTS PASSED"
    )
    print(f"Tests passed: {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
