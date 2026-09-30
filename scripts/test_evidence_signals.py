from __future__ import annotations

from src.chunking.chunk import Chunk
from src.diagnosis.evidence_signals import (
    extract_lexical_evidence_signals,
    informative_query_tokens,
    tokenize,
)
from src.retrieval.reranker import (
    RerankedRetrievalResult,
)


def make_chunk(
    chunk_id: str,
    document_id: str,
    text: str,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        text=text,
        start_char=0,
        end_char=len(text),
        metadata={},
    )


def main() -> None:

    query = (
        "Which university did the author "
        "of Novel X attend?"
    )

    results = [
        RerankedRetrievalResult(
            rank=1,
            score=8.0,
            chunk=make_chunk(
                "chunk_a",
                "doc_A",
                (
                    "Novel X was written by "
                    "a famous author."
                ),
            ),
            dense_rank=1,
            sparse_rank=1,
        ),
        RerankedRetrievalResult(
            rank=2,
            score=6.0,
            chunk=make_chunk(
                "chunk_b",
                "doc_B",
                (
                    "The author later became "
                    "a professor."
                ),
            ),
            dense_rank=2,
            sparse_rank=3,
        ),
        RerankedRetrievalResult(
            rank=3,
            score=4.0,
            chunk=make_chunk(
                "chunk_c",
                "doc_C",
                (
                    "The university is located "
                    "in Boston."
                ),
            ),
            dense_rank=3,
            sparse_rank=2,
        ),
    ]

    # ---------------------------------------------------------
    # Tokenization checks
    # ---------------------------------------------------------

    tokens = tokenize(query)

    informative = informative_query_tokens(
        query
    )

    print("All query tokens:")
    print(tokens)

    print("\nInformative query tokens:")
    print(informative)

    assert tokens == [
        "which",
        "university",
        "did",
        "the",
        "author",
        "of",
        "novel",
        "x",
        "attend",
    ]

    assert informative == [
        "university",
        "author",
        "novel",
        "x",
        "attend",
    ]

    # ---------------------------------------------------------
    # Signal extraction
    # ---------------------------------------------------------

    signals = (
        extract_lexical_evidence_signals(
            query=query,
            results=results,
        )
    )

    print("\nLexical evidence signals")
    print("------------------------")
    print(signals)

    # ---------------------------------------------------------
    # Query counts
    # ---------------------------------------------------------

    assert signals.query_token_count == 9

    assert (
        signals.informative_query_token_count
        == 5
    )

    # ---------------------------------------------------------
    # Coverage
    # ---------------------------------------------------------

    assert (
        signals.covered_query_token_count
        == 4
    )

    assert (
        abs(
            signals.query_token_coverage
            - 0.8
        )
        < 1e-9
    )

    assert (
        signals.uncovered_query_token_count
        == 1
    )

    # ---------------------------------------------------------
    # Repetition
    #
    # Only "author" appears in >= 2 documents.
    # ---------------------------------------------------------

    assert (
        signals.repeated_query_token_count
        == 1
    )

    assert (
        abs(
            signals.repeated_query_token_fraction
            - 0.2
        )
        < 1e-9
    )

    # ---------------------------------------------------------
    # Document frequency
    #
    # university = 1
    # author     = 2
    # novel      = 1
    # x          = 1
    # attend     = 0
    #
    # mean = 1.0
    # min  = 0
    # max  = 2
    # ---------------------------------------------------------

    assert (
        abs(
            signals.mean_query_token_document_frequency
            - 1.0
        )
        < 1e-9
    )

    assert (
        signals.min_query_token_document_frequency
        == 0
    )

    assert (
        signals.max_query_token_document_frequency
        == 2
    )

    # ---------------------------------------------------------
    # Empty informative-query edge case
    # ---------------------------------------------------------

    empty_signals = (
        extract_lexical_evidence_signals(
            query="the and of",
            results=results,
        )
    )

    assert (
        empty_signals.informative_query_token_count
        == 0
    )

    assert (
        empty_signals.query_token_coverage
        is None
    )

    assert (
        empty_signals.repeated_query_token_fraction
        is None
    )

    assert (
        empty_signals.mean_query_token_document_frequency
        is None
    )

    assert (
        empty_signals.min_query_token_document_frequency
        is None
    )

    assert (
        empty_signals.max_query_token_document_frequency
        is None
    )

    print()
    print(
        "All lexical evidence signal "
        "checks passed."
    )


if __name__ == "__main__":
    main()