from __future__ import annotations

import numpy as np

from src.chunking.chunk import Chunk
from src.diagnosis.evidence_signals import (
    extract_semantic_evidence_signals,
)
from src.embeddings.dense_embedder import (
    DenseEmbedder,
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


def make_result(
    rank: int,
    score: float,
    chunk: Chunk,
) -> RerankedRetrievalResult:
    return RerankedRetrievalResult(
        rank=rank,
        score=score,
        chunk=chunk,
        dense_rank=rank,
        sparse_rank=rank,
    )


def main() -> None:

    print("=" * 80)
    print(
        "PHASE 9.4E — SEMANTIC EVIDENCE "
        "SIGNAL TEST"
    )
    print("=" * 80)

    query = (
        "Which university did the author "
        "of Novel X attend?"
    )

    chunks = [
        make_chunk(
            chunk_id="chunk_a",
            document_id="doc_A",
            text=(
                "Novel X was written by "
                "a famous author."
            ),
        ),
        make_chunk(
            chunk_id="chunk_b",
            document_id="doc_B",
            text=(
                "The author attended "
                "Boston University."
            ),
        ),
        make_chunk(
            chunk_id="chunk_c",
            document_id="doc_C",
            text=(
                "The university was founded "
                "in the nineteenth century."
            ),
        ),
    ]

    results = [
        make_result(
            rank=1,
            score=8.0,
            chunk=chunks[0],
        ),
        make_result(
            rank=2,
            score=6.0,
            chunk=chunks[1],
        ),
        make_result(
            rank=3,
            score=4.0,
            chunk=chunks[2],
        ),
    ]

    print("\nLoading dense embedder...")

    embedder = DenseEmbedder()

    # ---------------------------------------------------------
    # Extract signals using production implementation
    # ---------------------------------------------------------

    signals = (
        extract_semantic_evidence_signals(
            query=query,
            results=results,
            embedder=embedder,
        )
    )

    print("\nSemantic evidence signals")
    print("-------------------------")
    print(signals)

    # ---------------------------------------------------------
    # Independently calculate expected values
    # ---------------------------------------------------------

    query_embedding = (
        embedder.encode_query(
            query
        )
    )

    chunk_embeddings = (
        embedder.encode_documents(
            [
                chunk.text
                for chunk in chunks
            ]
        )
    )

    query_similarities = (
        chunk_embeddings
        @ query_embedding
    )

    expected_max = float(
        np.max(
            query_similarities
        )
    )

    expected_mean = float(
        np.mean(
            query_similarities
        )
    )

    expected_spread = (
        expected_max
        - expected_mean
    )

    similarity_matrix = (
        chunk_embeddings
        @ chunk_embeddings.T
    )

    upper_triangle = (
        np.triu_indices(
            len(chunks),
            k=1,
        )
    )

    expected_pairwise = float(
        np.mean(
            similarity_matrix[
                upper_triangle
            ]
        )
    )

    # ---------------------------------------------------------
    # Compare production calculation with independent
    # calculation.
    # ---------------------------------------------------------

    tolerance = 1e-6

    assert signals.max_query_chunk_similarity is not None

    assert (
        signals.mean_query_chunk_similarity
        is not None
    )

    assert (
        signals.semantic_coverage_spread
        is not None
    )

    assert (
        signals.mean_pairwise_chunk_similarity
        is not None
    )

    assert abs(
        signals.max_query_chunk_similarity
        - expected_max
    ) < tolerance

    assert abs(
        signals.mean_query_chunk_similarity
        - expected_mean
    ) < tolerance

    assert abs(
        signals.semantic_coverage_spread
        - expected_spread
    ) < tolerance

    assert abs(
        signals.mean_pairwise_chunk_similarity
        - expected_pairwise
    ) < tolerance

    # ---------------------------------------------------------
    # Mathematical invariants
    # ---------------------------------------------------------

    assert (
        signals.max_query_chunk_similarity
        >= signals.mean_query_chunk_similarity
    )

    assert abs(
        signals.semantic_coverage_spread
        - (
            signals.max_query_chunk_similarity
            - signals.mean_query_chunk_similarity
        )
    ) < tolerance

    # ---------------------------------------------------------
    # Single-result case
    #
    # Query/chunk similarities are defined, but there are no
    # unique chunk pairs.
    # ---------------------------------------------------------

    single_signals = (
        extract_semantic_evidence_signals(
            query=query,
            results=[results[0]],
            embedder=embedder,
        )
    )

    assert (
        single_signals.max_query_chunk_similarity
        is not None
    )

    assert (
        single_signals.mean_query_chunk_similarity
        is not None
    )

    assert (
        single_signals.semantic_coverage_spread
        is not None
    )

    assert (
        single_signals.mean_pairwise_chunk_similarity
        is None
    )

    assert abs(
        single_signals.semantic_coverage_spread
    ) < tolerance

    # ---------------------------------------------------------
    # Empty retrieval case
    # ---------------------------------------------------------

    empty_signals = (
        extract_semantic_evidence_signals(
            query=query,
            results=[],
            embedder=embedder,
        )
    )

    assert (
        empty_signals.max_query_chunk_similarity
        is None
    )

    assert (
        empty_signals.mean_query_chunk_similarity
        is None
    )

    assert (
        empty_signals.semantic_coverage_spread
        is None
    )

    assert (
        empty_signals.mean_pairwise_chunk_similarity
        is None
    )

    print()
    print(
        "All semantic evidence signal "
        "checks passed."
    )


if __name__ == "__main__":
    main()