
"""Phase 11.2A tests for gold-free evidence comparison."""

from types import SimpleNamespace

from src.recovery.evidence_assessment import EvidenceAssessmentBaseline


def make_evidence(chunk_id: str, document_id: str, text: str):
    return SimpleNamespace(
        chunk=SimpleNamespace(
            id=chunk_id,
            document_id=document_id,
            text=text,
        )
    )


def test_coverage_gain():
    query = "Which city hosted the 2012 Olympics?"

    original = [
        make_evidence("a", "doc1", "The 2012 Olympics were held.")
    ]
    candidate = [
        make_evidence("a", "doc1", "The 2012 Olympics were held."),
        make_evidence("b", "doc2", "London is the host city."),
    ]

    comparison = EvidenceAssessmentBaseline().compare(
        query, original, candidate
    )

    assert "city" in comparison.gained_query_tokens
    assert comparison.coverage_delta > 0
    assert comparison.new_document_count == 1
    assert comparison.retained_chunk_count == 1


def test_coverage_loss():
    query = "Which city hosted the Olympics?"

    original = [
        make_evidence("a", "doc1", "The city hosted the Olympics.")
    ]
    candidate = [
        make_evidence("b", "doc2", "The Olympics were televised.")
    ]

    comparison = EvidenceAssessmentBaseline().compare(
        query, original, candidate
    )

    assert "city" in comparison.lost_query_tokens
    assert comparison.coverage_delta < 0
    assert comparison.retained_chunk_count == 0


def test_no_gold_required():
    comparison = EvidenceAssessmentBaseline().compare(
        "Who founded the company?",
        [make_evidence("a", "doc1", "The company was founded.")],
        [make_evidence("b", "doc2", "The founder was an engineer.")],
    )

    assert comparison.original.chunk_count == 1
    assert comparison.candidate.chunk_count == 1


def test_empty_informative_query():
    comparison = EvidenceAssessmentBaseline().compare(
        "the and of",
        [make_evidence("a", "doc1", "Some passage.")],
        [make_evidence("b", "doc2", "Another passage.")],
    )

    assert comparison.coverage_delta is None
    assert comparison.original.query_token_coverage is None
    assert comparison.candidate.query_token_coverage is None


def test_duplicate_documents_counted_once():
    comparison = EvidenceAssessmentBaseline().compare(
        "Which scientist discovered the element?",
        [make_evidence("a", "doc1", "Scientist biography.")],
        [
            make_evidence("b", "doc2", "Element discovery."),
            make_evidence("c", "doc2", "Scientific research."),
        ],
    )

    assert comparison.candidate.unique_document_count == 1
    assert comparison.new_document_count == 1


def main():
    tests = [
        test_coverage_gain,
        test_coverage_loss,
        test_no_gold_required,
        test_empty_informative_query,
        test_duplicate_documents_counted_once,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("\nPHASE 11.2A EVIDENCE ASSESSMENT TESTS PASSED")


if __name__ == "__main__":
    main()
