from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================================================
# Frozen experimental configuration
# =========================================================

ARTIFACT_DIR = Path("artifacts/diagnosis/scaled")

TRAIN_PATH = ARTIFACT_DIR / "diagnostic_train.csv"
TEST_PATH = ARTIFACT_DIR / "diagnostic_test.csv"

TARGET_COLUMN = "is_evidence_sufficient"
ORACLE_COLUMN = "oracle_failure_type"

FROZEN_THRESHOLD = 0.58


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


MODEL_FEATURES = (
    RETRIEVAL_FEATURES
    + LEXICAL_FEATURES
)


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

    required = (
        set(MODEL_FEATURES)
        | {
            TARGET_COLUMN,
            ORACLE_COLUMN,
            "split",
            "question_index",
        }
    )

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{path} missing columns: "
            f"{sorted(missing)}"
        )

    actual_splits = set(
        df["split"].astype(str).unique()
    )

    if actual_splits != {expected_split}:
        raise RuntimeError(
            f"Expected {expected_split!r}, "
            f"found {actual_splits}."
        )

    return df


# =========================================================
# Frozen model
# =========================================================

def build_frozen_model() -> Pipeline:

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
                    class_weight="balanced",
                ),
            ),
        ]
    )


# =========================================================
# Main
# =========================================================

def main() -> None:

    print("=" * 100)
    print(
        "PHASE 9.7E — FROZEN HELD-OUT "
        "SUFFICIENCY TEST EVALUATION"
    )
    print("=" * 100)

    print("\nFROZEN CONFIGURATION")
    print(
        f"Features: retrieval + lexical "
        f"({len(MODEL_FEATURES)})"
    )
    print(
        "Model: balanced logistic regression"
    )
    print(
        f"Threshold: {FROZEN_THRESHOLD:.2f}"
    )

    # -----------------------------------------------------
    # Load train and test only
    # -----------------------------------------------------

    print(
        "\nLoading frozen train and "
        "held-out test datasets..."
    )

    train_df = load_split(
        TRAIN_PATH,
        expected_split="train",
    )

    test_df = load_split(
        TEST_PATH,
        expected_split="test",
    )

    print(
        f"Train rows: {len(train_df):,}"
    )
    print(
        f"Test rows: {len(test_df):,}"
    )

    # Structural checks.
    if len(train_df) != 1000:
        raise RuntimeError(
            "Expected exactly 1,000 train rows."
        )

    if len(test_df) != 500:
        raise RuntimeError(
            "Expected exactly 500 test rows."
        )

    if test_df["question_index"].duplicated().any():
        raise RuntimeError(
            "Duplicate test question indices."
        )

    # -----------------------------------------------------
    # Train frozen detector
    # -----------------------------------------------------

    X_train = train_df[
        MODEL_FEATURES
    ].copy()

    y_train = train_df[
        TARGET_COLUMN
    ].astype(int).to_numpy()

    X_test = test_df[
        MODEL_FEATURES
    ].copy()

    y_test = test_df[
        TARGET_COLUMN
    ].astype(int).to_numpy()

    print(
        "\nTraining frozen detector on "
        "the 1,000-question training split..."
    )

    model = build_frozen_model()

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    # -----------------------------------------------------
    # Exactly one frozen test prediction
    # -----------------------------------------------------

    sufficient_probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    predicted_sufficient = (
        sufficient_probability
        >= FROZEN_THRESHOLD
    ).astype(int)

    # Operational positive class = insufficient.
    y_test_insufficient = 1 - y_test
    predicted_insufficient = (
        1 - predicted_sufficient
    )

    insufficient_probability = (
        1.0 - sufficient_probability
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predicted_sufficient,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            predicted_sufficient,
        )
    )

    insufficient_precision = (
        precision_score(
            y_test_insufficient,
            predicted_insufficient,
            zero_division=0,
        )
    )

    insufficient_recall = (
        recall_score(
            y_test_insufficient,
            predicted_insufficient,
            zero_division=0,
        )
    )

    insufficient_f1 = (
        f1_score(
            y_test_insufficient,
            predicted_insufficient,
            zero_division=0,
        )
    )

    # These rank the operational failure class.
    roc_auc = roc_auc_score(
        y_test_insufficient,
        insufficient_probability,
    )

    pr_auc = average_precision_score(
        y_test_insufficient,
        insufficient_probability,
    )

    matrix = confusion_matrix(
        y_test,
        predicted_sufficient,
        labels=[0, 1],
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

    correct_sufficient = int(
        matrix[1, 1]
    )

    recovery_trigger_rate = (
        predicted_insufficient.mean()
    )

    false_recovery_rate = (
        false_recovery_flags
        / int((y_test == 1).sum())
    )

    # -----------------------------------------------------
    # Main report
    # -----------------------------------------------------

    print("\n" + "=" * 100)
    print("HELD-OUT TEST RESULTS")
    print("=" * 100)

    print(
        f"\nTest sufficient / insufficient: "
        f"{int(y_test.sum())} / "
        f"{int((1 - y_test).sum())}"
    )

    print(
        f"\nAccuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"Insufficient precision: "
        f"{insufficient_precision:.4f}"
    )

    print(
        f"Insufficient recall: "
        f"{insufficient_recall:.4f}"
    )

    print(
        f"Insufficient F1: "
        f"{insufficient_f1:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{roc_auc:.4f}"
    )

    print(
        f"PR-AUC: "
        f"{pr_auc:.4f}"
    )

    print(
        f"\nDetected insufficient: "
        f"{detected_insufficient}"
    )

    print(
        f"Missed insufficient: "
        f"{missed_insufficient}"
    )

    print(
        f"False recovery flags: "
        f"{false_recovery_flags}"
    )

    print(
        f"Correct sufficient: "
        f"{correct_sufficient}"
    )

    print(
        f"Recovery trigger rate: "
        f"{recovery_trigger_rate:.4f}"
    )

    print(
        f"False recovery rate among "
        f"sufficient cases: "
        f"{false_recovery_rate:.4f}"
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        "Rows = true "
        "[insufficient, sufficient]"
    )

    print(
        "Cols = predicted "
        "[insufficient, sufficient]"
    )

    print(matrix)

    # -----------------------------------------------------
    # Oracle failure-type analysis
    # -----------------------------------------------------

    analysis_df = test_df.copy()

    analysis_df[
        "predicted_sufficient"
    ] = predicted_sufficient

    failures = analysis_df[
        analysis_df[
            TARGET_COLUMN
        ] == 0
    ].copy()

    rows = []

    for failure_type, group in (
        failures.groupby(
            ORACLE_COLUMN,
            sort=True,
        )
    ):
        total = len(group)

        detected = int(
            (
                group[
                    "predicted_sufficient"
                ] == 0
            ).sum()
        )

        missed = total - detected

        recall = (
            detected / total
            if total
            else np.nan
        )

        rows.append(
            {
                "oracle_failure_type": (
                    failure_type
                ),
                "total": total,
                "detected": detected,
                "missed": missed,
                "detection_recall": recall,
            }
        )

    failure_results = pd.DataFrame(
        rows
    )

    failure_results[
        "detection_recall"
    ] = failure_results[
        "detection_recall"
    ].round(4)

    print("\n" + "=" * 100)
    print(
        "HELD-OUT FAILURE DETECTION "
        "BY ORACLE TYPE"
    )
    print("=" * 100)

    print()
    print(
        failure_results.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Final statement
    # -----------------------------------------------------

    print("\n" + "=" * 100)
    print("PHASE 9.7E COMPLETE")
    print("=" * 100)

    print(
        "\nThis was the frozen held-out "
        "test evaluation."
    )

    print(
        "Features, model configuration, "
        "class weighting, and threshold "
        "were fixed before test evaluation."
    )

    print(
        "No test-based threshold search "
        "or model selection was performed."
    )

    print(
        "\nThe test results must now be "
        "reported as observed."
    )

    print(
        "Do not modify the binary detector "
        "based on these test results."
    )


if __name__ == "__main__":
    main()