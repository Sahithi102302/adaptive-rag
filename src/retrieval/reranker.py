from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sentence_transformers import CrossEncoder

from src.chunking.chunk import Chunk
from src.retrieval.dense_retriever import (
    DenseRetriever,
    format_chunk_for_retrieval,
)
from src.retrieval.sparse_retriever import BM25Retriever
from src.embeddings.dense_embedder import DenseEmbedder


@dataclass(frozen=True)
class RerankerConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidate_k: int = 20
    batch_size: int = 32


@dataclass(frozen=True)
class RerankedRetrievalResult:
    rank: int
    score: float
    chunk: Chunk

    dense_rank: int | None = None
    sparse_rank: int | None = None


@dataclass
class _Candidate:
    chunk: Chunk
    dense_rank: int | None = None
    sparse_rank: int | None = None


class CrossEncoderReranker:
    """
    Retrieve candidates independently with dense and BM25 retrieval,
    deduplicate their union, then rerank candidates using a
    query-passage cross-encoder.

    The cross-encoder is a relevance reranker. It should not be
    interpreted as an evidence-sufficiency model.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: BM25Retriever,
        config: RerankerConfig | None = None,
    ) -> None:

        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever

        self.config = (
            config
            if config is not None
            else RerankerConfig()
        )

        if self.config.candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than 0."
            )

        if self.config.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0."
            )

        self.model = CrossEncoder(
            self.config.model_name
        )

    def _build_candidate_pool(
        self,
        query: str,
        embedder: DenseEmbedder,
    ) -> list[_Candidate]:

        dense_results = self.dense_retriever.search(
            query=query,
            embedder=embedder,
            top_k=self.config.candidate_k,
        )

        sparse_results = self.sparse_retriever.search(
            query=query,
            top_k=self.config.candidate_k,
        )

        candidates: dict[str, _Candidate] = {}

        for result in dense_results:

            chunk_id = result.chunk.id

            if chunk_id not in candidates:
                candidates[chunk_id] = _Candidate(
                    chunk=result.chunk,
                )

            candidates[
                chunk_id
            ].dense_rank = result.rank

        for result in sparse_results:

            chunk_id = result.chunk.id

            if chunk_id not in candidates:
                candidates[chunk_id] = _Candidate(
                    chunk=result.chunk,
                )

            candidates[
                chunk_id
            ].sparse_rank = result.rank

        return list(candidates.values())

    def search(
        self,
        query: str,
        embedder: DenseEmbedder,
        top_k: int = 10,
    ) -> list[RerankedRetrievalResult]:

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        candidates = self._build_candidate_pool(
            query=query,
            embedder=embedder,
        )

        if not candidates:
            return []

        pairs: Sequence[tuple[str, str]] = [
            (
                query,
                format_chunk_for_retrieval(
                    candidate.chunk
                ),
            )
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )

        scored_candidates = [
            (
                float(score),
                candidate,
            )
            for score, candidate
            in zip(
                scores,
                candidates,
            )
        ]

        # Deterministic tie-breaking using chunk ID.
        scored_candidates.sort(
            key=lambda item: (
                -item[0],
                item[1].chunk.id,
            )
        )

        actual_k = min(
            top_k,
            len(scored_candidates),
        )

        results: list[
            RerankedRetrievalResult
        ] = []

        for rank, (
            score,
            candidate,
        ) in enumerate(
            scored_candidates[:actual_k],
            start=1,
        ):

            results.append(
                RerankedRetrievalResult(
                    rank=rank,
                    score=score,
                    chunk=candidate.chunk,
                    dense_rank=(
                        candidate.dense_rank
                    ),
                    sparse_rank=(
                        candidate.sparse_rank
                    ),
                )
            )

        return results