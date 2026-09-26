from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


EvidenceKey = Tuple[str, int]


class FailureFamily(str, Enum):
    """
    Broad layer of the RAG pipeline in which a failure occurs.
    """

    RETRIEVAL = "retrieval"
    EVIDENCE = "evidence"
    GENERATION = "generation"
    NONE = "none"


class FailureType(str, Enum):
    """
    Failure taxonomy used by Failure-Aware RAG.

    Retrieval failures are the first experimentally evaluated
    states. Other states are represented now so the architecture
    can later support ambiguity, temporal/conflict failures, and
    generation failures without mixing them with retrieval errors.
    """

    # ---------------------------------------------------------
    # No failure
    # ---------------------------------------------------------

    SUFFICIENT_EVIDENCE = "sufficient_evidence"

    # ---------------------------------------------------------
    # Retrieval-level failures
    # ---------------------------------------------------------

    SEVERE_RETRIEVAL_FAILURE = (
        "severe_retrieval_failure"
    )

    DOCUMENT_SELECTION_FAILURE = (
        "document_selection_failure"
    )

    PASSAGE_SELECTION_FAILURE = (
        "passage_selection_failure"
    )

    EVIDENCE_COVERAGE_FAILURE = (
        "evidence_coverage_failure"
    )

    # ---------------------------------------------------------
    # Evidence / query failures
    # ---------------------------------------------------------

    AMBIGUOUS_QUERY = "ambiguous_query"

    CONFLICTING_EVIDENCE = (
        "conflicting_evidence"
    )

    TEMPORAL_FAILURE = "temporal_failure"

    # ---------------------------------------------------------
    # Generation-level failures
    # ---------------------------------------------------------

    UNSUPPORTED_INFERENCE = (
        "unsupported_inference"
    )

    # ---------------------------------------------------------
    # Fallback
    # ---------------------------------------------------------

    UNKNOWN = "unknown"


class DiagnosisSource(str, Enum):
    """
    Describes where a diagnosis came from.

    ORACLE:
        Uses benchmark ground truth and is available only for
        offline evaluation.

    PREDICTED:
        Produced from signals available at inference time.

    NONE:
        No diagnosis was required.
    """

    ORACLE = "oracle"
    PREDICTED = "predicted"
    NONE = "none"


@dataclass(frozen=True)
class DiagnosticSignals:
    """
    Structured signals associated with one retrieval attempt.

    IMPORTANT:
    Some fields below are oracle-only because they depend on
    benchmark gold evidence. They must not later be used as
    inputs to the deployed online detector.
    """

    # ---------------------------------------------------------
    # Retrieval size
    # ---------------------------------------------------------

    retrieved_chunk_count: int

    # ---------------------------------------------------------
    # Oracle-only evidence statistics
    # ---------------------------------------------------------

    gold_fact_count: Optional[int] = None

    retrieved_gold_fact_count: Optional[int] = None

    gold_document_count: Optional[int] = None

    retrieved_gold_document_count: Optional[int] = None

    # ---------------------------------------------------------
    # Oracle-only structural signals
    # ---------------------------------------------------------

    has_any_gold_evidence: Optional[bool] = None

    has_partial_gold_evidence: Optional[bool] = None

    has_complete_gold_evidence: Optional[bool] = None

    has_missing_gold_document: Optional[bool] = None

    has_gold_document_wrong_passage: Optional[bool] = None

    # ---------------------------------------------------------
    # Future online signals
    # ---------------------------------------------------------

    top_score: Optional[float] = None

    score_margin: Optional[float] = None

    evidence_sufficiency_score: Optional[float] = None

    # ---------------------------------------------------------
    # Extensibility
    # ---------------------------------------------------------

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class Diagnosis:
    """
    Final structured diagnosis for one retrieval attempt.

    This object can represent either:
      - an oracle diagnosis during benchmark evaluation, or
      - a predicted diagnosis during real inference.
    """

    failure_type: FailureType

    failure_family: FailureFamily

    source: DiagnosisSource

    confidence: Optional[float]

    signals: DiagnosticSignals

    explanation: Optional[str] = None

    recovery_recommended: bool = False


def failure_family_for_type(
    failure_type: FailureType,
) -> FailureFamily:
    """
    Map a fine-grained failure type to its broader pipeline
    failure family.
    """

    if failure_type == FailureType.SUFFICIENT_EVIDENCE:
        return FailureFamily.NONE

    if failure_type in {
        FailureType.SEVERE_RETRIEVAL_FAILURE,
        FailureType.DOCUMENT_SELECTION_FAILURE,
        FailureType.PASSAGE_SELECTION_FAILURE,
        FailureType.EVIDENCE_COVERAGE_FAILURE,
    }:
        return FailureFamily.RETRIEVAL

    if failure_type in {
        FailureType.AMBIGUOUS_QUERY,
        FailureType.CONFLICTING_EVIDENCE,
        FailureType.TEMPORAL_FAILURE,
    }:
        return FailureFamily.EVIDENCE

    if failure_type == FailureType.UNSUPPORTED_INFERENCE:
        return FailureFamily.GENERATION

    return FailureFamily.NONE