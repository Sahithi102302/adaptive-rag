from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Tuple

from src.retrieval.reranker import (
    RerankedRetrievalResult,
)


class RecoveryAction(str, Enum):
    """
    Retrieval-side recovery actions supported by the
    failure-aware RAG pipeline.

    These values describe interventions, not failure labels.

    Runtime failure diagnosis and recovery action selection are
    deliberately kept separate because Phase 9 did not establish
    reliable fine-grained failure attribution.
    """

    NONE = "none"

    EXPANDED_CANDIDATE_RETRIEVAL = (
        "expanded_candidate_retrieval"
    )

    QUERY_REFORMULATION = (
        "query_reformulation"
    )

    MULTI_QUERY_RETRIEVAL = (
        "multi_query_retrieval"
    )


class RecoveryStatus(str, Enum):
    """
    Execution status of one recovery attempt.

    This describes whether the recovery operation executed,
    not whether it actually repaired the evidence failure.

    Recovery effectiveness is evaluated separately.
    """

    NOT_ATTEMPTED = "not_attempted"

    COMPLETED = "completed"

    FAILED = "failed"


@dataclass(frozen=True)
class RecoveryRequest:
    """
    Input to one recovery attempt.

    original_query:
        User/query text that produced the initial retrieval.

    action:
        Recovery intervention to execute.

    original_results:
        Initial reranked evidence. This is retained so later
        verification can compare evidence before and after recovery.

    metadata:
        Optional runtime information used by the recovery layer.
        Gold supporting facts must not be placed here during
        deployed/runtime execution.
    """

    original_query: str

    action: RecoveryAction

    original_results: Tuple[
        RerankedRetrievalResult,
        ...
    ]

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class RecoveryResult:
    """
    Output of one recovery attempt.

    The result deliberately preserves both the original evidence
    and recovered evidence.

    Phase 10 produces recovery candidates. It does NOT decide that
    recovered evidence is better simply because recovery executed.

    Phase 11 will perform post-recovery verification.
    """

    action: RecoveryAction

    status: RecoveryStatus

    original_query: str

    recovery_queries: Tuple[
        str,
        ...
    ]

    original_results: Tuple[
        RerankedRetrievalResult,
        ...
    ]

    recovered_results: Tuple[
        RerankedRetrievalResult,
        ...
    ]

    error_message: str | None = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def attempted(self) -> bool:
        """
        Return whether a recovery operation was attempted.
        """

        return self.status != RecoveryStatus.NOT_ATTEMPTED

    @property
    def succeeded(self) -> bool:
        """
        Return whether the recovery operation completed.

        IMPORTANT:
        completed execution does not imply that evidence quality
        improved.
        """

        return self.status == RecoveryStatus.COMPLETED