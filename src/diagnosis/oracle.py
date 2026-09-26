from __future__ import annotations

from typing import Sequence, Set, Tuple

from src.diagnosis.states import (
    Diagnosis,
    DiagnosisSource,
    DiagnosticSignals,
    EvidenceKey,
    FailureType,
    failure_family_for_type,
)
from src.retrieval.dense_retriever import (
    DenseRetrievalResult,
)


def extract_retrieved_evidence(
    results: Sequence[DenseRetrievalResult],
) -> Set[EvidenceKey]:
    """
    Extract every sentence-level evidence key represented by
    the retrieved sentence-aware chunks.
    """

    evidence: Set[EvidenceKey] = set()

    for result in results:
        chunk = result.chunk

        sentence_ids = chunk.metadata.get(
            "sentence_ids",
            [],
        )

        for sentence_id in sentence_ids:
            evidence.add(
                (
                    chunk.document_id,
                    sentence_id,
                )
            )

    return evidence


def extract_retrieved_documents(
    results: Sequence[DenseRetrievalResult],
) -> Set[str]:
    """
    Return the unique document IDs represented in retrieval.
    """

    return {
        result.chunk.document_id
        for result in results
    }


def diagnose_oracle_retrieval(
    gold_evidence: Set[EvidenceKey],
    results: Sequence[DenseRetrievalResult],
) -> Diagnosis:
    """
    Produce an OFFLINE oracle diagnosis using benchmark
    gold evidence.

    IMPORTANT:
    This function must never be used as the deployed online
    failure detector because it requires ground-truth evidence.

    Diagnostic precedence
    ---------------------

    1. Complete gold evidence
       -> SUFFICIENT_EVIDENCE

    2. No gold evidence
       -> SEVERE_RETRIEVAL_FAILURE

    3. All gold documents reached, but required gold
       sentences are missing
       -> PASSAGE_SELECTION_FAILURE

    4. Partial evidence retrieved, but one or more required
       gold documents are absent
       -> EVIDENCE_COVERAGE_FAILURE

    5. Required documents are absent without fitting the
       stronger evidence-coverage condition
       -> DOCUMENT_SELECTION_FAILURE

    6. Anything not captured above
       -> UNKNOWN

    These labels are benchmark-oriented oracle states.
    They are not yet the inference-time prediction rules.
    """

    if not gold_evidence:
        raise ValueError(
            "gold_evidence must contain at least "
            "one valid evidence key."
        )

    retrieved_evidence = (
        extract_retrieved_evidence(
            results
        )
    )

    retrieved_documents = (
        extract_retrieved_documents(
            results
        )
    )

    gold_documents = {
        document_id
        for document_id, _
        in gold_evidence
    }

    matched_gold_evidence = (
        gold_evidence
        & retrieved_evidence
    )

    retrieved_gold_documents = (
        gold_documents
        & retrieved_documents
    )

    missing_gold_evidence = (
        gold_evidence
        - retrieved_evidence
    )

    missing_gold_documents = (
        gold_documents
        - retrieved_documents
    )

    has_any_gold_evidence = (
        len(matched_gold_evidence) > 0
    )

    has_complete_gold_evidence = (
        gold_evidence.issubset(
            retrieved_evidence
        )
    )

    has_partial_gold_evidence = (
        has_any_gold_evidence
        and not has_complete_gold_evidence
    )

    has_missing_gold_document = (
        len(missing_gold_documents) > 0
    )

    has_gold_document_wrong_passage = any(
        document_id in retrieved_documents
        for document_id, _
        in missing_gold_evidence
    )

    # ---------------------------------------------------------
    # Retrieval-score signals
    # ---------------------------------------------------------

    top_score = None
    score_margin = None

    if results:
        top_score = float(
            results[0].score
        )

    if len(results) >= 2:
        score_margin = float(
            results[0].score
            - results[1].score
        )

    signals = DiagnosticSignals(
        retrieved_chunk_count=len(results),

        gold_fact_count=len(
            gold_evidence
        ),

        retrieved_gold_fact_count=len(
            matched_gold_evidence
        ),

        gold_document_count=len(
            gold_documents
        ),

        retrieved_gold_document_count=len(
            retrieved_gold_documents
        ),

        has_any_gold_evidence=(
            has_any_gold_evidence
        ),

        has_partial_gold_evidence=(
            has_partial_gold_evidence
        ),

        has_complete_gold_evidence=(
            has_complete_gold_evidence
        ),

        has_missing_gold_document=(
            has_missing_gold_document
        ),

        has_gold_document_wrong_passage=(
            has_gold_document_wrong_passage
        ),

        top_score=top_score,

        score_margin=score_margin,

        metadata={
            "missing_gold_fact_count": len(
                missing_gold_evidence
            ),
            "missing_gold_document_count": len(
                missing_gold_documents
            ),
        },
    )

    # ---------------------------------------------------------
    # State 1:
    # Complete evidence
    # ---------------------------------------------------------

    if has_complete_gold_evidence:

        failure_type = (
            FailureType.SUFFICIENT_EVIDENCE
        )

        explanation = (
            "All benchmark gold evidence facts "
            "are represented in the retrieved "
            "chunks."
        )

        recovery_recommended = False

    # ---------------------------------------------------------
    # State 2:
    # Severe retrieval failure
    # ---------------------------------------------------------

    elif not has_any_gold_evidence:

        failure_type = (
            FailureType.SEVERE_RETRIEVAL_FAILURE
        )

        explanation = (
            "No benchmark gold evidence fact "
            "is represented in the retrieved "
            "chunks."
        )

        recovery_recommended = True

    # ---------------------------------------------------------
    # State 3:
    # Correct documents reached, wrong passages
    # ---------------------------------------------------------

    elif (
        not has_missing_gold_document
        and has_gold_document_wrong_passage
    ):

        failure_type = (
            FailureType.PASSAGE_SELECTION_FAILURE
        )

        explanation = (
            "All required gold documents were "
            "reached, but at least one required "
            "gold sentence was not represented "
            "in the retrieved chunks."
        )

        recovery_recommended = True

    # ---------------------------------------------------------
    # State 4:
    # Partial evidence / missing document
    # ---------------------------------------------------------

    elif (
        has_partial_gold_evidence
        and has_missing_gold_document
    ):

        failure_type = (
            FailureType.EVIDENCE_COVERAGE_FAILURE
        )

        explanation = (
            "Some required evidence was retrieved, "
            "but at least one required gold "
            "document is missing."
        )

        recovery_recommended = True

    # ---------------------------------------------------------
    # State 5:
    # Document-selection failure
    # ---------------------------------------------------------

    elif has_missing_gold_document:

        failure_type = (
            FailureType.DOCUMENT_SELECTION_FAILURE
        )

        explanation = (
            "At least one required gold document "
            "is absent from the retrieved results."
        )

        recovery_recommended = True

    # ---------------------------------------------------------
    # Fallback
    # ---------------------------------------------------------

    else:

        failure_type = (
            FailureType.UNKNOWN
        )

        explanation = (
            "The retrieval attempt is incomplete "
            "but does not match the current oracle "
            "diagnostic rules."
        )

        recovery_recommended = True

    return Diagnosis(
        failure_type=failure_type,

        failure_family=(
            failure_family_for_type(
                failure_type
            )
        ),

        source=DiagnosisSource.ORACLE,

        # Oracle states are deterministic labels,
        # not probabilistic predictions.
        confidence=None,

        signals=signals,

        explanation=explanation,

        recovery_recommended=(
            recovery_recommended
        ),
    )