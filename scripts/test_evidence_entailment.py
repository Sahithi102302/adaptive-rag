
"""Phase 11.4E-1 evidence entailment contract tests."""

from dataclasses import replace

from src.recovery.evidence_entailment import (
    ChainEntailmentAssessment,
    ClaimVerification,
    EntailmentLabel,
)


def verification(
    label=EntailmentLabel.ENTAILED,
    citations=("book",),
):
    return ClaimVerification(
        claim="The Silver Atlas was written by Elena Marsh.",
        label=label,
        cited_chunk_ids=citations,
        explanation="Synthetic test assessment.",
        verifier_name="test_verifier",
    )


def assessment(*claims):
    return ChainEntailmentAssessment(
        claim_verifications=claims,
        assessed_chunk_ids=("book", "education"),
        explanation="Synthetic chain assessment.",
    )


def expect_error(error_type, operation):
    try:
        operation()
    except error_type:
        return
    raise AssertionError(
        f"Expected {error_type.__name__}."
    )


def test_entailed_claim():
    result = verification()
    assert result.label == EntailmentLabel.ENTAILED


def test_not_established_claim():
    result = verification(
        EntailmentLabel.NOT_ESTABLISHED
    )
    assert result.label == EntailmentLabel.NOT_ESTABLISHED


def test_contradicted_claim():
    result = assessment(
        verification(EntailmentLabel.CONTRADICTED)
    )
    assert result.has_contradiction
    assert not result.all_entailed


def test_all_entailed():
    result = assessment(
        verification(),
        replace(
            verification(),
            claim="Elena Marsh attended Northbridge University.",
            cited_chunk_ids=("education",),
        ),
    )
    assert result.all_entailed


def test_missing_claim():
    result = assessment(
        verification(),
        verification(
            EntailmentLabel.NOT_ESTABLISHED,
            ("education",),
        ),
    )
    assert result.has_unverified_claims
    assert not result.all_entailed


def test_unknown_claim():
    result = assessment(
        verification(EntailmentLabel.UNKNOWN, ())
    )
    assert result.has_unverified_claims


def test_empty_chain_not_entailed():
    result = assessment()
    assert not result.all_entailed


def test_unknown_citation_rejected():
    expect_error(
        ValueError,
        lambda: assessment(
            verification(
                EntailmentLabel.ENTAILED,
                ("invented",),
            )
        ),
    )


def test_duplicate_citation_rejected():
    expect_error(
        ValueError,
        lambda: verification(
            EntailmentLabel.ENTAILED,
            ("book", "book"),
        ),
    )


def test_definite_label_requires_citation():
    expect_error(
        ValueError,
        lambda: verification(
            EntailmentLabel.ENTAILED,
            (),
        ),
    )


def test_empty_claim_rejected():
    expect_error(
        ValueError,
        lambda: replace(
            verification(),
            claim="",
        ),
    )


def test_invalid_label_rejected():
    expect_error(
        TypeError,
        lambda: verification("entailed"),
    )


def test_invalid_verification_object_rejected():
    expect_error(
        TypeError,
        lambda: assessment("not a verification"),
    )


def test_duplicate_assessed_chunk_ids_rejected():
    expect_error(
        ValueError,
        lambda: replace(
            assessment(verification()),
            assessed_chunk_ids=("book", "book"),
        ),
    )


def test_empty_explanation_rejected():
    expect_error(
        ValueError,
        lambda: replace(
            verification(),
            explanation="",
        ),
    )


def main():
    tests = [
        test_entailed_claim,
        test_not_established_claim,
        test_contradicted_claim,
        test_all_entailed,
        test_missing_claim,
        test_unknown_claim,
        test_empty_chain_not_entailed,
        test_unknown_citation_rejected,
        test_duplicate_citation_rejected,
        test_definite_label_requires_citation,
        test_empty_claim_rejected,
        test_invalid_label_rejected,
        test_invalid_verification_object_rejected,
        test_duplicate_assessed_chunk_ids_rejected,
        test_empty_explanation_rejected,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(
        "\nPHASE 11.4E-1 EVIDENCE ENTAILMENT "
        "CONTRACT TESTS PASSED"
    )
    print(f"Tests passed: {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
