from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================================================
# Configuration
# =========================================================

ARTIFACT_DIR = Path("artifacts/diagnosis/scaled")

TRAIN_PATH = ARTIFACT_DIR / "diagnostic_train.csv"
VALIDATION_PATH = ARTIFACT_DIR / "diagnostic_validation.csv"

TARGET_COLUMN = "is_evidence_sufficient"


# =========================================================
# Feature families
# =========================================================

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


SEMANTIC_FEATURES = [
    "max_query_chunk_similarity",
    "mean_query_chunk_similarity",
    "semantic_coverage_spread",
    "mean_pairwise_chunk_similarity",
]


FEATURE_SETS = {
    "retrieval_only": (
        RETRIEVAL_FEATURES
    ),
    "retrieval_plus_lexical": (
        RETRIEVAL_FEATURES
        + LEXICAL_FEATURES
    ),
    "retrieval_plus_lexical_plus_semantic": (
        RETRIEVAL_FEATURES
        + LEXICAL_FEATURES
        + SEMANTIC_FEATURES
    ),
}


# =========================================================
# Loading / validation
# =========================================================

def load_split(
    path: Path,
    expected_split: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if "split" not in df.columns:
        raise RuntimeError(
            f"{path} does not contain a split column."
        )

    actual_splits = set(
        df["split"]
        .astype(str)
        .unique()
    )

    if actual_splits != {expected_split}:
        raise RuntimeError(
            f"Expected split {expected_split!r}, "
            f"found {actual_splits}."
        )

    return df


def validate_schema(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> None:
    all_model_features = set(
        RETRIEVAL_FEATURES
        + LEXICAL_FEATURES
        + SEMANTIC_FEATURES
    )

    required = (
        all_model_features
        | {
            TARGET_COLUMN,
            "split",
        }
    )

    for name, df in (
        ("train", train_df),
        ("validation", validation_df),
    ):
        missing = required - set(df.columns)

        if missing:
            raise RuntimeError(
                f"{name} dataset is missing "
                f"columns: {sorted(missing)}"
            )

    # Explicit leakage guard.
    forbidden = {
        "question",
        "question_index",
        "split",
        "oracle_failure_type",
        TARGET_COLUMN,
    }

    leaked = (
        all_model_features
        & forbidden
    )

    if leaked:
        raise RuntimeError(
            "Forbidden feature leakage detected: "
            f"{sorted(leaked)}"
        )


# =========================================================
# Model
# =========================================================

def build_model(
    class_weight,
) -> Pipeline:
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
                    max_iter=2000,
                    random_state=42,
                    class_weight=class_weight,
                ),
            ),
        ]
    )


# =========================================================
# Metrics
# =========================================================

def evaluate_model(
    y_true: np.ndarray,
    sufficient_probability: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    y_pred = (
        sufficient_probability >= threshold
    ).astype(int)

    # Operational failure-detection perspective:
    # 1 = insufficient
    # 0 = sufficient
    y_true_insufficient = 1 - y_true
    y_pred_insufficient = 1 - y_pred

    insufficient_probability = (
        1.0 - sufficient_probability
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    true_insufficient = int(
        (y_true == 0).sum()
    )

    predicted_insufficient = int(
        (y_pred == 0).sum()
    )

    detected_insufficient = int(
        matrix[0, 0]
    )

    missed_insufficient = int(
        matrix[0, 1]
    )

    false_recovery_flags = int(
        matrix[1, 0]
    )

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "balanced_accuracy": (
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "insufficient_precision": (
            precision_score(
                y_true_insufficient,
                y_pred_insufficient,
                zero_division=0,
            )
        ),
        "insufficient_recall": (
            recall_score(
                y_true_insufficient,
                y_pred_insufficient,
                zero_division=0,
            )
        ),
        "insufficient_f1": (
            f1_score(
                y_true_insufficient,
                y_pred_insufficient,
                zero_division=0,
            )
        ),
        "roc_auc": roc_auc_score(
            y_true_insufficient,
            insufficient_probability,
        ),
        "pr_auc": average_precision_score(
            y_true_insufficient,
            insufficient_probability,
        ),
        "true_insufficient": (
            true_insufficient
        ),
        "predicted_insufficient": (
            predicted_insufficient
        ),
        "detected_insufficient": (
            detected_insufficient
        ),
        "missed_insufficient": (
            missed_insufficient
        ),
        "false_recovery_flags": (
            false_recovery_flags
        ),
        "confusion_matrix": matrix,
    }


# =========================================================
# Reporting
# =========================================================

def print_single_result(
    experiment_name: str,
    feature_count: int,
    metrics: dict,
) -> None:
    print("\n" + "=" * 90)
    print(experiment_name)
    print("=" * 90)

    print(
        f"Feature count: {feature_count}"
    )

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Insufficient precision: "
        f"{metrics['insufficient_precision']:.4f}"
    )

    print(
        f"Insufficient recall: "
        f"{metrics['insufficient_recall']:.4f}"
    )

    print(
        f"Insufficient F1: "
        f"{metrics['insufficient_f1']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC: "
        f"{metrics['pr_auc']:.4f}"
    )

    print(
        "\nFailure-detection counts:"
    )

    print(
        f"Actual insufficient: "
        f"{metrics['true_insufficient']}"
    )

    print(
        f"Detected insufficient: "
        f"{metrics['detected_insufficient']}"
    )

    print(
        f"Missed insufficient: "
        f"{metrics['missed_insufficient']}"
    )

    print(
        f"False recovery flags: "
        f"{metrics['false_recovery_flags']}"
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        "Rows = true [insufficient, sufficient]\n"
        "Cols = predicted [insufficient, sufficient]"
    )

    print(
        metrics["confusion_matrix"]
    )


# =========================================================
# Main
# =========================================================

def main() -> None:
    print("=" * 90)
    print(
        "PHASE 9.7A — CONTROLLED "
        "EVIDENCE-FEATURE ABLATION"
    )
    print("=" * 90)

    # ---------------------------------------------------------
    # Load frozen datasets
    # ---------------------------------------------------------

    print(
        "\nLoading frozen datasets..."
    )

    train_df = load_split(
        TRAIN_PATH,
        expected_split="train",
    )

    validation_df = load_split(
        VALIDATION_PATH,
        expected_split="validation",
    )

    validate_schema(
        train_df,
        validation_df,
    )

    print(
        f"Train rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    y_train = train_df[
        TARGET_COLUMN
    ].astype(int).to_numpy()

    y_validation = validation_df[
        TARGET_COLUMN
    ].astype(int).to_numpy()

    print(
        "\nTrain sufficient / insufficient: "
        f"{int(y_train.sum()):,} / "
        f"{int((1 - y_train).sum()):,}"
    )

    print(
        "Validation sufficient / insufficient: "
        f"{int(y_validation.sum()):,} / "
        f"{int((1 - y_validation).sum()):,}"
    )

    # ---------------------------------------------------------
    # Controlled experiment
    # ---------------------------------------------------------

    model_configs = {
        "plain": None,
        "balanced": "balanced",
    }

    result_rows = []

    for feature_set_name, features in (
        FEATURE_SETS.items()
    ):
        print("\n" + "#" * 90)

        print(
            f"FEATURE SET: {feature_set_name}"
        )

        print(
            f"Feature count: {len(features)}"
        )

        print("#" * 90)

        X_train = train_df[
            features
        ].copy()

        X_validation = validation_df[
            features
        ].copy()

        for model_name, class_weight in (
            model_configs.items()
        ):
            print(
                f"\nTraining {model_name} model..."
            )

            pipeline = build_model(
                class_weight=class_weight,
            )

            pipeline.fit(
                X_train,
                y_train,
            )

            validation_probability = (
                pipeline.predict_proba(
                    X_validation
                )[:, 1]
            )

            metrics = evaluate_model(
                y_true=y_validation,
                sufficient_probability=(
                    validation_probability
                ),
                threshold=0.5,
            )

            experiment_name = (
                f"{feature_set_name} | "
                f"{model_name}"
            )

            print_single_result(
                experiment_name=(
                    experiment_name
                ),
                feature_count=len(features),
                metrics=metrics,
            )

            result_rows.append(
                {
                    "feature_set": (
                        feature_set_name
                    ),
                    "model": model_name,
                    "feature_count": (
                        len(features)
                    ),
                    "accuracy": (
                        metrics["accuracy"]
                    ),
                    "balanced_accuracy": (
                        metrics[
                            "balanced_accuracy"
                        ]
                    ),
                    "insufficient_precision": (
                        metrics[
                            "insufficient_precision"
                        ]
                    ),
                    "insufficient_recall": (
                        metrics[
                            "insufficient_recall"
                        ]
                    ),
                    "insufficient_f1": (
                        metrics[
                            "insufficient_f1"
                        ]
                    ),
                    "roc_auc": (
                        metrics["roc_auc"]
                    ),
                    "pr_auc": (
                        metrics["pr_auc"]
                    ),
                    "detected_insufficient": (
                        metrics[
                            "detected_insufficient"
                        ]
                    ),
                    "missed_insufficient": (
                        metrics[
                            "missed_insufficient"
                        ]
                    ),
                    "false_recovery_flags": (
                        metrics[
                            "false_recovery_flags"
                        ]
                    ),
                }
            )

    # ---------------------------------------------------------
    # Final comparison table
    # ---------------------------------------------------------

    results_df = pd.DataFrame(
        result_rows
    )

    print("\n" + "=" * 120)
    print(
        "VALIDATION ABLATION SUMMARY"
    )
    print("=" * 120)

    display_columns = [
        "feature_set",
        "model",
        "feature_count",
        "accuracy",
        "balanced_accuracy",
        "insufficient_precision",
        "insufficient_recall",
        "insufficient_f1",
        "roc_auc",
        "pr_auc",
        "detected_insufficient",
        "missed_insufficient",
        "false_recovery_flags",
    ]

    display_df = results_df[
        display_columns
    ].copy()

    numeric_columns = [
        "accuracy",
        "balanced_accuracy",
        "insufficient_precision",
        "insufficient_recall",
        "insufficient_f1",
        "roc_auc",
        "pr_auc",
    ]

    display_df[
        numeric_columns
    ] = display_df[
        numeric_columns
    ].round(4)

    print()
    print(
        display_df.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Explicit deltas vs retrieval-only
    # ---------------------------------------------------------

    print("\n" + "=" * 120)
    print(
        "DELTA VS RETRIEVAL-ONLY "
        "(SAME MODEL CONFIGURATION)"
    )
    print("=" * 120)

    delta_metrics = [
        "balanced_accuracy",
        "insufficient_recall",
        "insufficient_f1",
        "roc_auc",
        "pr_auc",
    ]

    for model_name in model_configs:
        print(
            f"\nModel: {model_name}"
        )

        baseline_row = results_df[
            (
                results_df["feature_set"]
                == "retrieval_only"
            )
            & (
                results_df["model"]
                == model_name
            )
        ].iloc[0]

        for feature_set_name in [
            "retrieval_plus_lexical",
            (
                "retrieval_plus_lexical"
                "_plus_semantic"
            ),
        ]:
            candidate_row = results_df[
                (
                    results_df["feature_set"]
                    == feature_set_name
                )
                & (
                    results_df["model"]
                    == model_name
                )
            ].iloc[0]

            print(
                f"\n  {feature_set_name}"
            )

            for metric in delta_metrics:
                delta = (
                    candidate_row[metric]
                    - baseline_row[metric]
                )

                print(
                    f"    {metric:<24} "
                    f"{delta:+.4f}"
                )

    # ---------------------------------------------------------
    # Completion
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print(
        "PHASE 9.7A COMPLETE"
    )
    print("=" * 90)

    print(
        "\nThis experiment compares feature "
        "representations under identical model settings."
    )

    print(
        "No threshold tuning was performed."
    )

    print(
        "No test examples were loaded or evaluated."
    )

    print(
        "The 500-question test split remains untouched."
    )


if __name__ == "__main__":
    main()