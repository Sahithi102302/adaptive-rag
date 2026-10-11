from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from src.diagnosis.evidence_signals import extract_lexical_evidence_signals
from src.diagnosis.runtime import (
    RUNTIME_FEATURES,
    RuntimeDiagnosisResult,
    RuntimeSufficiencyDiagnoser,
)
from src.diagnosis.signals import extract_runtime_signals
from src.recovery.controller import ControllerResult, RecoveryController
from src.retrieval.reranker import RerankedRetrievalResult


@dataclass(frozen=True)
class DiagnosedControllerResult:
    feature_row: dict[str, object]
    diagnosis: RuntimeDiagnosisResult
    controller_result: ControllerResult


def build_runtime_feature_row(
    query: str,
    results: Sequence[RerankedRetrievalResult],
) -> dict[str, object]:
    """Reproduce the frozen Phase 9 retrieval + lexical feature inputs.

    Never uses semantic features, gold supporting facts, or oracle labels.
    """
    retrieval_features = asdict(extract_runtime_signals(results))
    lexical_features = asdict(
        extract_lexical_evidence_signals(query=query, results=results)
    )
    overlap = set(retrieval_features) & set(lexical_features)
    if overlap:
        raise ValueError(f"Overlapping Phase 9 feature names: {sorted(overlap)}")
    combined = {**retrieval_features, **lexical_features}
    expected = set(RUNTIME_FEATURES)
    missing = expected - set(combined)
    if missing:
        raise ValueError(f"Missing frozen Phase 9 features: {sorted(missing)}")
    # Explicit ordering matches the model's training/inference column ordering.
    return {name: combined[name] for name in RUNTIME_FEATURES}


class DiagnosedRecoveryController:
    """Run the fitted Phase 9 detector, then the Phase 10.5A controller.

    Fitting/loading the frozen detector is deliberately a separate concern.
    No recovery output is accepted before the future verification gate.
    """

    def __init__(
        self,
        diagnoser: RuntimeSufficiencyDiagnoser,
        controller: RecoveryController,
    ) -> None:
        self.diagnoser = diagnoser
        self.controller = controller

    def run(
        self,
        query: str,
        original_results: Sequence[RerankedRetrievalResult],
    ) -> DiagnosedControllerResult:
        original = tuple(original_results)
        features = build_runtime_feature_row(query=query, results=original)
        diagnosis = self.diagnoser.diagnose(features)
        decision = self.controller.run(
            query=query, original_results=original, diagnosis=diagnosis
        )
        return DiagnosedControllerResult(
            feature_row=features,
            diagnosis=diagnosis,
            controller_result=decision,
        )
