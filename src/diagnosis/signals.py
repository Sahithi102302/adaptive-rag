from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean
from typing import Sequence

from src.retrieval.reranker import RerankedRetrievalResult


@dataclass(frozen=True)
class RuntimeDiagnosticSignals:
    """
    Signals that are available at inference time.

    IMPORTANT:
    This structure must never contain benchmark gold evidence,
    gold document IDs, gold sentence IDs, or oracle diagnoses.

    These features describe only the query's retrieved results.
    """

    # ---------------------------------------------------------
    # Retrieval size
    # ---------------------------------------------------------

    retrieved_chunk_count: int
    unique_document_count: int

    # ---------------------------------------------------------
    # Cross-encoder score statistics
    # ---------------------------------------------------------

    top_score: float | None
    second_score: float | None
    score_margin: float | None

    mean_score: float | None
    score_std: float | None
    min_score: float | None
    max_score: float | None

    # ---------------------------------------------------------
    # Document concentration
    # ---------------------------------------------------------

    max_chunks_per_document: int
    max_document_fraction: float | None

    # ---------------------------------------------------------
    # Dense / sparse retrieval agreement
    # ---------------------------------------------------------

    both_retriever_count: int
    dense_only_count: int
    sparse_only_count: int

    both_retriever_fraction: float | None
    dense_only_fraction: float | None
    sparse_only_fraction: float | None

    # ---------------------------------------------------------
    # Rank agreement
    # ---------------------------------------------------------

    mean_rank_disagreement: float | None
    max_rank_disagreement: int | None


def _population_std(values: Sequence[float]) -> float | None:
    """
    Compute population standard deviation.

    Returns None for an empty sequence.
    """

    if not values:
        return None

    value_mean = mean(values)

    variance = mean(
        (value - value_mean) ** 2
        for value in values
    )

    return sqrt(variance)


def extract_runtime_signals(
    results: Sequence[RerankedRetrievalResult],
) -> RuntimeDiagnosticSignals:
    """
    Extract diagnostic features using only information available
    from reranked retrieval results.

    No benchmark ground truth is accessed here.
    """

    retrieved_chunk_count = len(results)

    # ---------------------------------------------------------
    # Empty retrieval
    # ---------------------------------------------------------

    if not results:
        return RuntimeDiagnosticSignals(
            retrieved_chunk_count=0,
            unique_document_count=0,

            top_score=None,
            second_score=None,
            score_margin=None,

            mean_score=None,
            score_std=None,
            min_score=None,
            max_score=None,

            max_chunks_per_document=0,
            max_document_fraction=None,

            both_retriever_count=0,
            dense_only_count=0,
            sparse_only_count=0,

            both_retriever_fraction=None,
            dense_only_fraction=None,
            sparse_only_fraction=None,

            mean_rank_disagreement=None,
            max_rank_disagreement=None,
        )

    # ---------------------------------------------------------
    # Cross-encoder score statistics
    # ---------------------------------------------------------

    scores = [
        float(result.score)
        for result in results
    ]

    top_score = scores[0]

    second_score = (
        scores[1]
        if len(scores) >= 2
        else None
    )

    score_margin = (
        top_score - second_score
        if second_score is not None
        else None
    )

    # ---------------------------------------------------------
    # Document diversity / concentration
    # ---------------------------------------------------------

    document_counts: dict[str, int] = {}

    for result in results:
        document_id = result.chunk.document_id

        document_counts[document_id] = (
            document_counts.get(document_id, 0) + 1
        )

    unique_document_count = len(document_counts)

    max_chunks_per_document = max(
        document_counts.values()
    )

    max_document_fraction = (
        max_chunks_per_document
        / retrieved_chunk_count
    )

    # ---------------------------------------------------------
    # Dense / sparse agreement
    # ---------------------------------------------------------

    both_retriever_count = 0
    dense_only_count = 0
    sparse_only_count = 0

    rank_disagreements: list[int] = []

    for result in results:

        has_dense = result.dense_rank is not None
        has_sparse = result.sparse_rank is not None

        if has_dense and has_sparse:
            both_retriever_count += 1

            rank_disagreements.append(
                abs(
                    result.dense_rank
                    - result.sparse_rank
                )
            )

        elif has_dense:
            dense_only_count += 1

        elif has_sparse:
            sparse_only_count += 1

    denominator = float(retrieved_chunk_count)

    both_retriever_fraction = (
        both_retriever_count / denominator
    )

    dense_only_fraction = (
        dense_only_count / denominator
    )

    sparse_only_fraction = (
        sparse_only_count / denominator
    )

    # ---------------------------------------------------------
    # Rank disagreement
    # ---------------------------------------------------------

    if rank_disagreements:

        mean_rank_disagreement = mean(
            rank_disagreements
        )

        max_rank_disagreement = max(
            rank_disagreements
        )

    else:

        mean_rank_disagreement = None
        max_rank_disagreement = None

    # ---------------------------------------------------------
    # Final runtime feature object
    # ---------------------------------------------------------

    return RuntimeDiagnosticSignals(
        retrieved_chunk_count=(
            retrieved_chunk_count
        ),

        unique_document_count=(
            unique_document_count
        ),

        top_score=top_score,
        second_score=second_score,
        score_margin=score_margin,

        mean_score=mean(scores),

        score_std=_population_std(scores),

        min_score=min(scores),
        max_score=max(scores),

        max_chunks_per_document=(
            max_chunks_per_document
        ),

        max_document_fraction=(
            max_document_fraction
        ),

        both_retriever_count=(
            both_retriever_count
        ),

        dense_only_count=(
            dense_only_count
        ),

        sparse_only_count=(
            sparse_only_count
        ),

        both_retriever_fraction=(
            both_retriever_fraction
        ),

        dense_only_fraction=(
            dense_only_fraction
        ),

        sparse_only_fraction=(
            sparse_only_fraction
        ),

        mean_rank_disagreement=(
            mean_rank_disagreement
        ),

        max_rank_disagreement=(
            max_rank_disagreement
        ),
    )