from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.chunking.chunk import Chunk
from src.embeddings.dense_embedder import DenseEmbedder
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import BM25Retriever


@dataclass(frozen=True)
class HybridRetrievalConfig:
    """
    Configuration for Reciprocal Rank Fusion.

    candidate_k:
        Number of candidates requested independently from
        dense and BM25 retrieval before fusion.

    rrf_k:
        Stabilizing constant used in the RRF equation.
    """

    candidate_k: int = 20
    rrf_k: int = 60


@dataclass(frozen=True)
class HybridRetrievalResult:
    """
    One result after dense + BM25 Reciprocal Rank Fusion.
    """

    rank: int
    score: float
    chunk: Chunk

    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None


class HybridRetriever:
    """
    Hybrid dense + sparse retriever using Reciprocal Rank Fusion.

    Raw dense cosine scores and BM25 scores are deliberately
    not combined because they live on different numerical scales.

    Instead, each retriever contributes according to rank:

        1 / (rrf_k + rank)
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: BM25Retriever,
        config: HybridRetrievalConfig | None = None,
    ) -> None:

        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever

        self.config = (
            config
            if config is not None
            else HybridRetrievalConfig()
        )

        if self.config.candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than 0."
            )

        if self.config.rrf_k <= 0:
            raise ValueError(
                "rrf_k must be greater than 0."
            )

    def search(
        self,
        query: str,
        embedder: DenseEmbedder,
        top_k: int = 10,
    ) -> list[HybridRetrievalResult]:
        """
        Retrieve dense and sparse candidate sets, fuse them
        with RRF, and return the final top-k chunks.
        """

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        candidate_k = max(
            top_k,
            self.config.candidate_k,
        )

        dense_results = (
            self.dense_retriever.search(
                query=query,
                embedder=embedder,
                top_k=candidate_k,
            )
        )

        sparse_results = (
            self.sparse_retriever.search(
                query=query,
                top_k=candidate_k,
            )
        )

        # -----------------------------------------------------
        # Accumulate candidates by stable chunk ID.
        # -----------------------------------------------------

        candidates: dict[str, dict] = {}

        for result in dense_results:

            chunk_id = result.chunk.id

            candidates.setdefault(
                chunk_id,
                {
                    "chunk": result.chunk,
                    "score": 0.0,
                    "dense_rank": None,
                    "sparse_rank": None,
                },
            )

            candidates[chunk_id][
                "dense_rank"
            ] = result.rank

            candidates[chunk_id][
                "score"
            ] += (
                1.0
                / (
                    self.config.rrf_k
                    + result.rank
                )
            )

        for result in sparse_results:

            chunk_id = result.chunk.id

            candidates.setdefault(
                chunk_id,
                {
                    "chunk": result.chunk,
                    "score": 0.0,
                    "dense_rank": None,
                    "sparse_rank": None,
                },
            )

            candidates[chunk_id][
                "sparse_rank"
            ] = result.rank

            candidates[chunk_id][
                "score"
            ] += (
                1.0
                / (
                    self.config.rrf_k
                    + result.rank
                )
            )

        # -----------------------------------------------------
        # Deterministic ranking.
        #
        # Primary:
        #   higher RRF score
        #
        # Tie breaker:
        #   stable chunk ID
        # -----------------------------------------------------

        ranked_candidates = sorted(
            candidates.values(),
            key=lambda item: (
                -item["score"],
                item["chunk"].id,
            ),
        )

        actual_k = min(
            top_k,
            len(ranked_candidates),
        )

        results: list[
            HybridRetrievalResult
        ] = []

        for rank, item in enumerate(
            ranked_candidates[:actual_k],
            start=1,
        ):

            results.append(
                HybridRetrievalResult(
                    rank=rank,
                    score=float(
                        item["score"]
                    ),
                    chunk=item["chunk"],
                    dense_rank=item[
                        "dense_rank"
                    ],
                    sparse_rank=item[
                        "sparse_rank"
                    ],
                )
            )

        return results
    