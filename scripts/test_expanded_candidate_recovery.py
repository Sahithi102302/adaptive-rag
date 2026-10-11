from __future__ import annotations

import numpy as np

from src.chunking.chunk import Chunk
from src.embeddings.dense_embedder import DenseEmbedder
from src.recovery.expanded_candidates import (
    ExpandedCandidateRecovery,
    ExpandedCandidateRecoveryConfig,
)
from src.recovery.states import (
    RecoveryAction,
    RecoveryRequest,
    RecoveryStatus,
)
from src.retrieval.dense_retriever import (
    DenseRetriever,
    format_chunk_for_retrieval,
)
from src.retrieval.reranker import (
    CrossEncoderReranker,
    RerankerConfig,
)
from src.retrieval.sparse_retriever import BM25Retriever


def make_chunk(
    chunk_id: str,
    title: str,
    text: str,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=f"doc_{chunk_id}",
        chunk_index=0,
        text=text,
        start_char=0,
        end_char=len(text),
        metadata={
            "title": title,
        },
    )


def main() -> None:

    print("=" * 100)
    print(
        "PHASE 10.2A — EXPANDED CANDIDATE "
        "RECOVERY TEST"
    )
    print("=" * 100)

    chunks = [
        make_chunk(
            "c1",
            "Paris",
            "Paris is the capital city of France.",
        ),
        make_chunk(
            "c2",
            "France",
            "France is a country in Western Europe.",
        ),
        make_chunk(
            "c3",
            "Berlin",
            "Berlin is the capital city of Germany.",
        ),
        make_chunk(
            "c4",
            "Germany",
            "Germany is a country in Central Europe.",
        ),
        make_chunk(
            "c5",
            "Madrid",
            "Madrid is the capital city of Spain.",
        ),
        make_chunk(
            "c6",
            "Spain",
            "Spain is a country in Southwestern Europe.",
        ),
    ]

    embedder = DenseEmbedder()

    retrieval_texts = [
        format_chunk_for_retrieval(chunk)
        for chunk in chunks
    ]

    embeddings = embedder.encode_documents(
        retrieval_texts
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    dense_retriever = DenseRetriever(
        chunks=chunks,
        embeddings=embeddings,
    )

    sparse_retriever = BM25Retriever(
        chunks=chunks,
        retrieval_texts=retrieval_texts,
    )

    query = "What is the capital of France?"

    baseline_reranker = CrossEncoderReranker(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        config=RerankerConfig(
            candidate_k=2,
            batch_size=8,
        ),
    )

    original_results = baseline_reranker.search(
        query=query,
        embedder=embedder,
        top_k=2,
    )

    request = RecoveryRequest(
        original_query=query,
        action=(
            RecoveryAction
            .EXPANDED_CANDIDATE_RETRIEVAL
        ),
        original_results=tuple(
            original_results
        ),
        metadata={
            "test": "phase_10_2a",
        },
    )

    recovery = ExpandedCandidateRecovery(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        embedder=embedder,
        config=ExpandedCandidateRecoveryConfig(
            candidate_k=5,
            final_top_k=2,
        ),
        reranker_batch_size=8,
    )

    result = recovery.recover(
        request
    )

    assert result.status == RecoveryStatus.COMPLETED
    assert result.succeeded is True

    assert result.original_query == query

    assert result.recovery_queries == (
        query,
    )

    assert (
        result.original_results
        == tuple(original_results)
    )

    assert len(result.recovered_results) == 2

    assert result.metadata[
        "candidate_k"
    ] == 5

    assert result.metadata[
        "final_top_k"
    ] == 2

    assert (
        recovery.reranker.config.candidate_k
        == 5
    )

    assert (
        baseline_reranker.config.candidate_k
        == 2
    )

    print()
    print(
        "Baseline candidate_k:",
        baseline_reranker.config.candidate_k,
    )

    print(
        "Recovery candidate_k:",
        recovery.reranker.config.candidate_k,
    )

    print()
    print("Original top results:")

    for item in result.original_results:
        print(
            f"  rank={item.rank} "
            f"score={item.score:.4f} "
            f"chunk={item.chunk.id}"
        )

    print()
    print("Recovered top results:")

    for item in result.recovered_results:
        print(
            f"  rank={item.rank} "
            f"score={item.score:.4f} "
            f"chunk={item.chunk.id}"
        )

    print()
    print(
        "Recovery status:",
        result.status.value,
    )

    print(
        "Execution succeeded:",
        result.succeeded,
    )

    print()
    print("=" * 100)
    print(
        "PHASE 10.2A EXPANDED CANDIDATE "
        "RECOVERY TEST PASSED"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()