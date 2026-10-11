from __future__ import annotations

from dataclasses import dataclass

from src.embeddings.dense_embedder import DenseEmbedder
from src.recovery.states import (
    RecoveryAction,
    RecoveryRequest,
    RecoveryResult,
    RecoveryStatus,
)
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.reranker import (
    CrossEncoderReranker,
    RerankerConfig,
)
from src.retrieval.sparse_retriever import BM25Retriever


@dataclass(frozen=True)
class ExpandedCandidateRecoveryConfig:
    """
    Configuration for expanded-candidate retrieval recovery.

    candidate_k:
        Number of candidates requested independently from
        dense and sparse retrieval before cross-encoder
        reranking.

    final_top_k:
        Number of reranked chunks returned after recovery.

    The final evidence depth should normally match the original
    retrieval depth so before/after evidence remains comparable.
    """

    candidate_k: int = 40
    final_top_k: int = 10

    def __post_init__(self) -> None:
        if self.candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than 0."
            )

        if self.final_top_k <= 0:
            raise ValueError(
                "final_top_k must be greater than 0."
            )

        if self.candidate_k < self.final_top_k:
            raise ValueError(
                "candidate_k must be greater than or "
                "equal to final_top_k."
            )


class ExpandedCandidateRecovery:
    """
    Recovery strategy that broadens dense + sparse candidate
    retrieval while keeping the original query unchanged.

    A new CrossEncoderReranker is created with a larger candidate
    pool rather than mutating the baseline reranker's configuration.

    Completion of this recovery does not imply evidence quality
    improved. Phase 11 verification will make that determination.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: BM25Retriever,
        embedder: DenseEmbedder,
        config: ExpandedCandidateRecoveryConfig | None = None,
        reranker_model_name: str = (
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        reranker_batch_size: int = 32,
    ) -> None:

        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.embedder = embedder

        self.config = (
            config
            if config is not None
            else ExpandedCandidateRecoveryConfig()
        )

        if reranker_batch_size <= 0:
            raise ValueError(
                "reranker_batch_size must be greater than 0."
            )

        self.reranker = CrossEncoderReranker(
            dense_retriever=self.dense_retriever,
            sparse_retriever=self.sparse_retriever,
            config=RerankerConfig(
                model_name=reranker_model_name,
                candidate_k=self.config.candidate_k,
                batch_size=reranker_batch_size,
            ),
        )

    def recover(
        self,
        request: RecoveryRequest,
    ) -> RecoveryResult:
        """
        Execute expanded-candidate retrieval recovery.
        """

        if (
            request.action
            != RecoveryAction.EXPANDED_CANDIDATE_RETRIEVAL
        ):
            raise ValueError(
                "ExpandedCandidateRecovery received an "
                f"incompatible action: {request.action.value}"
            )

        try:
            recovered_results = self.reranker.search(
                query=request.original_query,
                embedder=self.embedder,
                top_k=self.config.final_top_k,
            )

            return RecoveryResult(
                action=request.action,
                status=RecoveryStatus.COMPLETED,
                original_query=request.original_query,
                recovery_queries=(
                    request.original_query,
                ),
                original_results=request.original_results,
                recovered_results=tuple(
                    recovered_results
                ),
                metadata={
                    **request.metadata,
                    "candidate_k": (
                        self.config.candidate_k
                    ),
                    "final_top_k": (
                        self.config.final_top_k
                    ),
                    "strategy": (
                        RecoveryAction
                        .EXPANDED_CANDIDATE_RETRIEVAL
                        .value
                    ),
                },
            )

        except Exception as exc:
            return RecoveryResult(
                action=request.action,
                status=RecoveryStatus.FAILED,
                original_query=request.original_query,
                recovery_queries=(
                    request.original_query,
                ),
                original_results=request.original_results,
                recovered_results=(),
                error_message=str(exc),
                metadata={
                    **request.metadata,
                    "candidate_k": (
                        self.config.candidate_k
                    ),
                    "final_top_k": (
                        self.config.final_top_k
                    ),
                    "strategy": (
                        RecoveryAction
                        .EXPANDED_CANDIDATE_RETRIEVAL
                        .value
                    ),
                },
            )