from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.embeddings.dense_embedder import DenseEmbedder
from src.retrieval.reranker import (
    CrossEncoderReranker,
    RerankedRetrievalResult,
)


@dataclass(frozen=True)
class SecondHopRetrievalConfig:
    """
    Configuration for controlled second-hop retrieval.

    top_k controls only the number of final reranked
    second-hop results returned.

    Candidate generation depth remains controlled by the
    frozen CrossEncoderReranker configuration.
    """

    top_k: int = 10

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )


@dataclass(frozen=True)
class SecondHopRetrievalResult:
    """
    Result of one second-hop retrieval attempt.

    The first-hop and second-hop result sets are preserved
    separately so later recovery evaluation can determine
    exactly what evidence was available before and after the
    intervention.

    No gold evidence or oracle information is stored here.
    """

    query: str

    first_hop_results: tuple[
        RerankedRetrievalResult,
        ...,
    ]

    second_hop_results: tuple[
        RerankedRetrievalResult,
        ...,
    ]


class SecondHopRetriever:
    """
    Execute one controlled second-hop retrieval using the
    existing frozen retrieval/reranking pipeline.

    This class performs retrieval only.

    It does NOT:
      - generate queries,
      - select bridge entities,
      - inspect gold evidence,
      - diagnose failures,
      - decide whether recovery succeeded,
      - merge or rerank first-hop and second-hop results.

    Those responsibilities remain separate.
    """

    def __init__(
        self,
        reranker: CrossEncoderReranker,
        embedder: DenseEmbedder,
        config: SecondHopRetrievalConfig | None = None,
    ) -> None:
        self.reranker = reranker
        self.embedder = embedder

        self.config = (
            config
            if config is not None
            else SecondHopRetrievalConfig()
        )

    def retrieve(
        self,
        query: str,
        first_hop_results: Sequence[
            RerankedRetrievalResult
        ],
    ) -> SecondHopRetrievalResult:
        """
        Execute second-hop retrieval for one generated query.

        first_hop_results are preserved exactly as supplied.
        They are not modified, merged, filtered, or reranked.
        """

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "Second-hop query must be non-empty."
            )

        preserved_first_hop = tuple(
            first_hop_results
        )

        second_hop_results = (
            self.reranker.search(
                query=normalized_query,
                embedder=self.embedder,
                top_k=self.config.top_k,
            )
        )

        return SecondHopRetrievalResult(
            query=normalized_query,
            first_hop_results=(
                preserved_first_hop
            ),
            second_hop_results=tuple(
                second_hop_results
            ),
        )