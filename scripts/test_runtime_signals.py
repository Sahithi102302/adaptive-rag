from __future__ import annotations

from src.chunking.chunk import Chunk
from src.diagnosis.signals import extract_runtime_signals
from src.retrieval.reranker import RerankedRetrievalResult


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

    results = [
        RerankedRetrievalResult(
            rank=1,
            score=8.0,
            chunk=make_chunk(
                "chunk_a1",
                "doc_A",
                "First passage from document A.",
            ),
            dense_rank=1,
            sparse_rank=2,
        ),

        RerankedRetrievalResult(
            rank=2,
            score=6.0,
            chunk=make_chunk(
                "chunk_a2",
                "doc_A",
                "Second passage from document A.",
            ),
            dense_rank=3,
            sparse_rank=4,
        ),

        RerankedRetrievalResult(
            rank=3,
            score=4.0,
            chunk=make_chunk(
                "chunk_b1",
                "doc_B",
                "Passage from document B.",
            ),
            dense_rank=5,
            sparse_rank=None,
        ),

        RerankedRetrievalResult(
            rank=4,
            score=2.0,
            chunk=make_chunk(
                "chunk_c1",
                "doc_C",
                "Passage from document C.",
            ),
            dense_rank=None,
            sparse_rank=6,
        ),
    ]

    signals = extract_runtime_signals(results)

    print("Runtime diagnostic signals")
    print("--------------------------")
    print(signals)

    # ---------------------------------------------------------
    # Retrieval/document checks
    # ---------------------------------------------------------

    assert signals.retrieved_chunk_count == 4
    assert signals.unique_document_count == 3

    # ---------------------------------------------------------
    # Score checks
    # ---------------------------------------------------------

    assert signals.top_score == 8.0
    assert signals.second_score == 6.0
    assert signals.score_margin == 2.0

    assert signals.mean_score == 5.0
    assert signals.min_score == 2.0
    assert signals.max_score == 8.0

    # Population std of [8, 6, 4, 2]
    expected_std = (5.0) ** 0.5

    assert abs(
        signals.score_std - expected_std
    ) < 1e-9

    # ---------------------------------------------------------
    # Document concentration
    # ---------------------------------------------------------

    assert signals.max_chunks_per_document == 2

    assert signals.max_document_fraction == 0.5

    # ---------------------------------------------------------
    # Retriever agreement
    # ---------------------------------------------------------

    assert signals.both_retriever_count == 2
    assert signals.dense_only_count == 1
    assert signals.sparse_only_count == 1

    assert signals.both_retriever_fraction == 0.5
    assert signals.dense_only_fraction == 0.25
    assert signals.sparse_only_fraction == 0.25

    # ---------------------------------------------------------
    # Rank disagreement
    # ---------------------------------------------------------

    assert signals.mean_rank_disagreement == 1.0
    assert signals.max_rank_disagreement == 1

    print()
    print("All runtime signal checks passed.")


if __name__ == "__main__":
    main()