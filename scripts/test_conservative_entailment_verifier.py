
"""Phase 11.4E-2 deterministic entailment baseline tests."""

from src.recovery.conservative_entailment_verifier import (
    ConservativeEntailmentVerifier,
)
from src.recovery.evidence_entailment import EntailmentLabel


VERIFIER = ConservativeEntailmentVerifier()

EVIDENCE = {
    "book": "The Silver Atlas was written by Elena Marsh.",
    "education": "Elena Marsh attended Northbridge University.",
    "other": "Daniel Reed attended Northbridge University.",
    "negative": "Elena Marsh never attended Northbridge University.",
}


def verify(claim, citations):
    return VERIFIER.verify(
        claim=claim,
        cited_chunk_ids=citations,
        evidence_by_id=EVIDENCE,
    )


def expect_error(error_type, operation):
    try:
        operation()
    except error_type:
        return

    raise AssertionError(
        f"Expected {error_type.__name__}."
    )


def test_explicit_author_claim():
    result = verify(
        "The Silver Atlas was written by Elena Marsh.",
        ("book",),
    )
    assert result.label == EntailmentLabel.ENTAILED


def test_explicit_education_claim():
    result = verify(
        "Elena Marsh attended Northbridge University.",
        ("education",),
    )
    assert result.label == EntailmentLabel.ENTAILED


def test_case_and_whitespace_normalization():
    result = verify(
        "  ELENA MARSH   attended Northbridge University! ",
        ("education",),
    )
    assert result.label == EntailmentLabel.ENTAILED


def test_invented_university():
    result = verify(
        "Elena Marsh attended the University of Cambridge.",
        ("book",),
    )
    assert result.label == EntailmentLabel.NOT_ESTABLISHED


def test_missing_first_hop():
    result = verify(
        "The Silver Atlas was written by Elena Marsh.",
        ("education",),
    )
    assert result.label == EntailmentLabel.NOT_ESTABLISHED


def test_disconnected_author():
    result = verify(
        "The Silver Atlas was written by Daniel Reed.",
        ("book",),
    )
    assert result.label == EntailmentLabel.NOT_ESTABLISHED


def test_negative_evidence_does_not_entail_positive():
    result = verify(
        "Elena Marsh attended Northbridge University.",
        ("negative",),
    )
    assert result.label == EntailmentLabel.NOT_ESTABLISHED


def test_positive_evidence_does_not_entail_negative():
    result = verify(
        "Elena Marsh never attended Northbridge University.",
        ("education",),
    )
    assert result.label == EntailmentLabel.NOT_ESTABLISHED


def test_no_citations():
    result = verify(
        "Elena Marsh attended Northbridge University.",
        (),
    )
    assert result.label == EntailmentLabel.UNKNOWN


def test_unknown_citation():
    expect_error(
        ValueError,
        lambda: verify(
            "Elena Marsh attended Northbridge University.",
            ("invented",),
        ),
    )


def test_duplicate_citations():
    expect_error(
        ValueError,
        lambda: verify(
            "Elena Marsh attended Northbridge University.",
            ("education", "education"),
        ),
    )


def test_paraphrase_not_accepted():
    result = verify(
        "Northbridge University was attended by Elena Marsh.",
        ("education",),
    )
    assert result.label == EntailmentLabel.NOT_ESTABLISHED


def test_substring_not_accepted():
    result = verify(
        "Elena Marsh",
        ("education",),
    )
    assert result.label == EntailmentLabel.NOT_ESTABLISHED


def test_claim_in_multisentence_passage():
    result = VERIFIER.verify(
        claim="Elena Marsh attended Northbridge University.",
        cited_chunk_ids=("combined",),
        evidence_by_id={
            "combined": (
                "The Silver Atlas was written by Elena Marsh. "
                "Elena Marsh attended Northbridge University."
            )
        },
    )
    assert result.label == EntailmentLabel.ENTAILED


def test_invalid_claim():
    expect_error(
        ValueError,
        lambda: verify("", ("book",)),
    )


def main():
    tests = [
        test_explicit_author_claim,
        test_explicit_education_claim,
        test_case_and_whitespace_normalization,
        test_invented_university,
        test_missing_first_hop,
        test_disconnected_author,
        test_negative_evidence_does_not_entail_positive,
        test_positive_evidence_does_not_entail_negative,
        test_no_citations,
        test_unknown_citation,
        test_duplicate_citations,
        test_paraphrase_not_accepted,
        test_substring_not_accepted,
        test_claim_in_multisentence_passage,
        test_invalid_claim,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(
        "\nPHASE 11.4E-2 CONSERVATIVE ENTAILMENT "
        "VERIFIER TESTS PASSED"
    )
    print(f"Tests passed: {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
