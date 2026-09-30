from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.diagnosis.states import (
    Diagnosis,
    DiagnosisSource,
    DiagnosticSignals,
    FailureFamily,
    FailureType,
)


# =========================================================
# Frozen Phase 9 detector configuration
# =========================================================

FROZEN_SUFFICIENCY_THRESHOLD = 0.58


RETRIEVAL_FEATURES = [
    "retrieved_chunk_count",
    "unique_document_count",
    "top_score",
    "second_score",
    "score_margin",
    "mean_score",
    "score_std",
    "min_score",
    "max_score",
    "max_chunks_per_document",
    "max_document_fraction",
    "both_retriever_count",
    "dense_only_count",
    "sparse_only_count",
    "both_retriever_fraction",
    "dense_only_fraction",
    "sparse_only_fraction",
    "mean_rank_disagreement",
    "max_rank_disagreement",
]


LEXICAL_FEATURES = [
    "query_token_count",
    "informative_query_token_count",
    "covered_query_token_count",
    "query_token_coverage",
    "repeated_query_token_count",
    "repeated_query_token_fraction",
    "mean_query_token_document_frequency",
    "min_query_token_document_frequency",
    "max_query_token_document_frequency",
    "uncovered_query_token_count",
]


RUNTIME_FEATURES = (
    RETRIEVAL_FEATURES
    + LEXICAL_FEATURES
)


# =========================================================
# Runtime result
# =========================================================

@dataclass(frozen=True)
class RuntimeDiagnosisResult:
    """
    Output of the learned runtime sufficiency detector.

    sufficient_probability:
        Logistic-regression model output P(sufficient).

    predicted_sufficient:
        True when P(sufficient) >= the configured threshold.

    The probability is a model output, not a guarantee that
    the retrieved evidence is actually sufficient.
    """

    diagnosis: Diagnosis
    sufficient_probability: float
    predicted_sufficient: bool


# =========================================================
# Runtime sufficiency diagnoser
# =========================================================

class RuntimeSufficiencyDiagnoser:
    """
    Frozen Phase 9 runtime evidence-sufficiency detector.

    The deployed Phase 9 detector is intentionally binary.

    Supported runtime decisions:
        - sufficient evidence
        - insufficient / uncertain evidence

    Fine-grained oracle labels such as:
        - evidence-coverage failure
        - passage-selection failure
        - severe-retrieval failure

    are NOT predicted here.

    Phase 9.8 found some fine-grained signal, but minority-class
    support was too small to justify deploying a reliable learned
    mechanism classifier.
    """

    def __init__(
        self,
        threshold: float = FROZEN_SUFFICIENCY_THRESHOLD,
    ) -> None:

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        self.threshold = float(threshold)

        self._model: Pipeline | None = None

    # -----------------------------------------------------
    # Model construction
    # -----------------------------------------------------

    @staticmethod
    def _build_model() -> Pipeline:
        """
        Construct the frozen Phase 9 detector specification.

        Configuration:
            - median imputation
            - standard scaling
            - class-balanced logistic regression
            - max_iter=2000
            - random_state=42
        """

        return Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        )

    # -----------------------------------------------------
    # State
    # -----------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """
        Return whether the detector has been fitted.
        """

        return self._model is not None

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    def fit(
        self,
        feature_frame: pd.DataFrame,
        labels: Sequence[int],
    ) -> "RuntimeSufficiencyDiagnoser":
        """
        Fit the frozen detector specification.

        labels:
            1 = evidence sufficient
            0 = evidence insufficient

        Only runtime-available features are accepted.

        Gold supporting facts and oracle failure labels must never
        be passed as detector inputs.
        """

        missing = [
            feature
            for feature in RUNTIME_FEATURES
            if feature not in feature_frame.columns
        ]

        if missing:
            raise ValueError(
                "Missing runtime features: "
                f"{missing}"
            )

        X = feature_frame[
            RUNTIME_FEATURES
        ].copy()

        y = np.asarray(
            labels,
            dtype=int,
        )

        if len(X) != len(y):
            raise ValueError(
                "Feature rows and labels have "
                "different lengths."
            )

        unique_labels = set(
            np.unique(y).tolist()
        )

        if not unique_labels.issubset(
            {0, 1}
        ):
            raise ValueError(
                "Labels must contain only 0 and 1."
            )

        if unique_labels != {0, 1}:
            raise ValueError(
                "Training requires both sufficient "
                "and insufficient examples."
            )

        self._model = self._build_model()

        self._model.fit(
            X,
            y,
        )

        return self

    # -----------------------------------------------------
    # Probability prediction
    # -----------------------------------------------------

    def predict_probability(
        self,
        feature_row: pd.Series | dict,
    ) -> float:
        """
        Return the fitted model's P(sufficient).

        This is the logistic-regression model output and should not
        be interpreted as calibrated diagnostic confidence.
        """

        if self._model is None:
            raise RuntimeError(
                "RuntimeSufficiencyDiagnoser "
                "must be fitted before prediction."
            )

        if isinstance(
            feature_row,
            pd.Series,
        ):
            row = feature_row.to_dict()
        else:
            row = dict(feature_row)

        missing = [
            feature
            for feature in RUNTIME_FEATURES
            if feature not in row
        ]

        if missing:
            raise ValueError(
                "Prediction row missing runtime "
                f"features: {missing}"
            )

        X = pd.DataFrame(
            [
                {
                    feature: row[feature]
                    for feature in RUNTIME_FEATURES
                }
            ]
        )

        probability = float(
            self._model.predict_proba(
                X
            )[0, 1]
        )

        return probability

    # -----------------------------------------------------
    # DiagnosticSignals construction
    # -----------------------------------------------------

    @staticmethod
    def _make_diagnostic_signals(
        feature_row: pd.Series | dict,
        sufficient_probability: float,
    ) -> DiagnosticSignals:
        """
        Convert runtime-only detector features into the stable
        DiagnosticSignals structure.

        DiagnosticSignals intentionally exposes only a small set of
        generic online fields directly.

        Richer Phase 9 detector-specific runtime features are stored
        in DiagnosticSignals.metadata.

        Oracle/gold fields are intentionally left as None.
        """

        if isinstance(
            feature_row,
            pd.Series,
        ):
            row = feature_row.to_dict()
        else:
            row = dict(feature_row)

        def optional_float(
            name: str,
        ) -> float | None:
            """
            Safely convert a possibly missing/NaN value to float.
            """

            value = row.get(name)

            if value is None or pd.isna(value):
                return None

            return float(value)

        # -------------------------------------------------
        # Detector-specific runtime metadata
        # -------------------------------------------------

        runtime_metadata = {
            # ---------------------------------------------
            # Retrieval structure
            # ---------------------------------------------

            "unique_document_count": int(
                row["unique_document_count"]
            ),

            "second_score": optional_float(
                "second_score"
            ),

            "mean_score": optional_float(
                "mean_score"
            ),

            "score_std": optional_float(
                "score_std"
            ),

            "min_score": optional_float(
                "min_score"
            ),

            "max_score": optional_float(
                "max_score"
            ),

            "max_chunks_per_document": int(
                row["max_chunks_per_document"]
            ),

            "max_document_fraction": optional_float(
                "max_document_fraction"
            ),

            # ---------------------------------------------
            # Dense / sparse retriever agreement
            # ---------------------------------------------

            "both_retriever_count": int(
                row["both_retriever_count"]
            ),

            "dense_only_count": int(
                row["dense_only_count"]
            ),

            "sparse_only_count": int(
                row["sparse_only_count"]
            ),

            "both_retriever_fraction": optional_float(
                "both_retriever_fraction"
            ),

            "dense_only_fraction": optional_float(
                "dense_only_fraction"
            ),

            "sparse_only_fraction": optional_float(
                "sparse_only_fraction"
            ),

            "mean_rank_disagreement": optional_float(
                "mean_rank_disagreement"
            ),

            "max_rank_disagreement": optional_float(
                "max_rank_disagreement"
            ),

            # ---------------------------------------------
            # Lexical evidence-coverage signals
            # ---------------------------------------------

            "query_token_count": int(
                row["query_token_count"]
            ),

            "informative_query_token_count": int(
                row[
                    "informative_query_token_count"
                ]
            ),

            "covered_query_token_count": int(
                row[
                    "covered_query_token_count"
                ]
            ),

            "query_token_coverage": optional_float(
                "query_token_coverage"
            ),

            "repeated_query_token_count": int(
                row[
                    "repeated_query_token_count"
                ]
            ),

            "repeated_query_token_fraction": optional_float(
                "repeated_query_token_fraction"
            ),

            "mean_query_token_document_frequency": optional_float(
                "mean_query_token_document_frequency"
            ),

            "min_query_token_document_frequency": optional_float(
                "min_query_token_document_frequency"
            ),

            "max_query_token_document_frequency": optional_float(
                "max_query_token_document_frequency"
            ),

            "uncovered_query_token_count": int(
                row[
                    "uncovered_query_token_count"
                ]
            ),

            # ---------------------------------------------
            # Detector provenance
            # ---------------------------------------------

            "detector_feature_count": len(
                RUNTIME_FEATURES
            ),

            "detector_version": (
                "phase_9_retrieval_lexical_lr"
            ),
        }

        # -------------------------------------------------
        # Stable DiagnosticSignals representation
        # -------------------------------------------------

        return DiagnosticSignals(
            retrieved_chunk_count=int(
                row["retrieved_chunk_count"]
            ),

            # Oracle-only fields intentionally remain None.

            top_score=optional_float(
                "top_score"
            ),

            score_margin=optional_float(
                "score_margin"
            ),

            evidence_sufficiency_score=float(
                sufficient_probability
            ),

            metadata=runtime_metadata,
        )

    # -----------------------------------------------------
    # Runtime diagnosis
    # -----------------------------------------------------

    def diagnose(
        self,
        feature_row: pd.Series | dict,
    ) -> RuntimeDiagnosisResult:
        """
        Produce a conservative runtime diagnosis.

        If:
            P(sufficient) >= threshold

        then:
            SUFFICIENT_EVIDENCE
            recovery_recommended = False

        Otherwise:
            UNKNOWN
            recovery_recommended = True

        UNKNOWN does not mean that the system observed no problem.

        It means that the learned binary detector predicts evidence
        insufficiency, while Phase 9 does not provide enough evidence
        to make a reliable fine-grained runtime attribution.
        """

        probability = self.predict_probability(
            feature_row
        )

        predicted_sufficient = (
            probability >= self.threshold
        )

        signals = (
            self._make_diagnostic_signals(
                feature_row=feature_row,
                sufficient_probability=probability,
            )
        )

        # -------------------------------------------------
        # Sufficient evidence
        # -------------------------------------------------

        if predicted_sufficient:

            diagnosis = Diagnosis(
                failure_type=(
                    FailureType.SUFFICIENT_EVIDENCE
                ),

                failure_family=(
                    FailureFamily.NONE
                ),

                source=(
                    DiagnosisSource.PREDICTED
                ),

                # The model probability has not been
                # calibrated as diagnosis confidence.
                confidence=None,

                signals=signals,

                explanation=(
                    "The frozen runtime sufficiency "
                    "detector classified the retrieved "
                    "evidence as sufficient. "
                    f"P(sufficient)={probability:.4f}, "
                    f"threshold={self.threshold:.2f}."
                ),

                recovery_recommended=False,
            )

        # -------------------------------------------------
        # Insufficient / uncertain evidence
        # -------------------------------------------------

        else:

            diagnosis = Diagnosis(
                failure_type=(
                    FailureType.UNKNOWN
                ),

                failure_family=(
                    FailureFamily.EVIDENCE
                ),

                source=(
                    DiagnosisSource.PREDICTED
                ),

                # Do not interpret raw P(sufficient)
                # as calibrated diagnosis confidence.
                confidence=None,

                signals=signals,

                explanation=(
                    "The frozen runtime sufficiency "
                    "detector classified the retrieved "
                    "evidence as insufficient or "
                    "uncertain. Fine-grained failure "
                    "attribution is intentionally left "
                    "unknown because Phase 9.8 did not "
                    "establish adequate support for a "
                    "reliable learned mechanism "
                    "classifier. "
                    f"P(sufficient)={probability:.4f}, "
                    f"threshold={self.threshold:.2f}."
                ),

                recovery_recommended=True,
            )

        return RuntimeDiagnosisResult(
            diagnosis=diagnosis,
            sufficient_probability=probability,
            predicted_sufficient=(
                predicted_sufficient
            ),
        )