
"""Phase 11.4G-2: natural-language question evidence diagnosis.

Connects conservative question decomposition with integrated
evidence coverage reporting.

This is a deterministic diagnostic pipeline, not a general
semantic verifier or an automatic answer acceptance system.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from src.recovery.question_decomposer import (
    ConservativeQuestionDecomposer,
    DecompositionStatus,
)
from src.recovery.integrated_coverage_reporter import (
    IntegratedCoverageReporter,
    IntegratedCoverageReport,
    IntegratedDecision,
)


class DiagnosisStatus(str, Enum):
    UNIQUE_CANDIDATE = "unique_candidate"
    AMBIGUOUS = "ambiguous"
    CONFLICTING = "conflicting"
    INCOMPLETE = "incomplete"
    UNSUPPORTED_QUESTION = "unsupported_question"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class QuestionEvidenceDiagnosis:
    question: str
    status: DiagnosisStatus
    candidate_answers: tuple[str, ...]
    clean_candidate_answers: tuple[str, ...]
    coverage_report: IntegratedCoverageReport | None
    explanation: str

    @property
    def eligible_for_automatic_acceptance(self) -> bool:
        return False


class QuestionEvidenceDiagnoser:
    name = "question_evidence_diagnoser_v1"

    _DECISION_MAP = {
        IntegratedDecision.UNIQUE_CANDIDATE:
            DiagnosisStatus.UNIQUE_CANDIDATE,
        IntegratedDecision.AMBIGUOUS:
            DiagnosisStatus.AMBIGUOUS,
        IntegratedDecision.CONFLICTING:
            DiagnosisStatus.CONFLICTING,
        IntegratedDecision.INCOMPLETE:
            DiagnosisStatus.INCOMPLETE,
        IntegratedDecision.UNKNOWN:
            DiagnosisStatus.UNKNOWN,
    }

    def __init__(self) -> None:
        self.decomposer = ConservativeQuestionDecomposer()
        self.reporter = IntegratedCoverageReporter()

    def diagnose(
        self,
        question: str,
        evidence_by_id: Mapping[str, str],
    ) -> QuestionEvidenceDiagnosis:
        if not isinstance(evidence_by_id, Mapping):
            raise TypeError(
                "Evidence must be a mapping of chunk IDs to text."
            )

        for chunk_id, passage in evidence_by_id.items():
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(
                    "Evidence chunk IDs must be nonempty strings."
                )
            if not isinstance(passage, str):
                raise TypeError(
                    "Evidence passages must be strings."
                )

        decomposition = self.decomposer.decompose(question)

        if decomposition.status != DecompositionStatus.SUPPORTED:
            return QuestionEvidenceDiagnosis(
                question=question,
                status=DiagnosisStatus.UNSUPPORTED_QUESTION,
                candidate_answers=(),
                clean_candidate_answers=(),
                coverage_report=None,
                explanation=decomposition.explanation,
            )

        requirements = decomposition.requirements

        if requirements is None:
            return QuestionEvidenceDiagnosis(
                question=question,
                status=DiagnosisStatus.UNKNOWN,
                candidate_answers=(),
                clean_candidate_answers=(),
                coverage_report=None,
                explanation=(
                    "Decomposition was marked supported but "
                    "did not provide requirements."
                ),
            )

        coverage = self.reporter.assess(
            requirements,
            evidence_by_id,
        )

        return QuestionEvidenceDiagnosis(
            question=question,
            status=self._DECISION_MAP[coverage.decision],
            candidate_answers=coverage.candidate_answers,
            clean_candidate_answers=(
                coverage.clean_candidate_answers
            ),
            coverage_report=coverage,
            explanation=coverage.explanation,
        )
