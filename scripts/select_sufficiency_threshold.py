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

# Predeclared operating constraint.
MIN_INSUFFICIENT_RECALL = 0.70

# We predict P(sufficient).
#
# If:
#     P(sufficient) >= threshold
# then:
#     sufficient
#
# Otherwise:
#     insufficient -> recovery candidate
#
# Because insufficient = probability below threshold,
# increasing this threshold generally makes the detector
# more willing to trigger recovery.
THRESHOLDS = np.round(
    np.arange(
        0.05,
        0.951,
        0.01,
    ),
    2,
)


# =========================================================
# Frozen feature representation
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


MODEL_FEATURES = (
    RETRIEVAL_FEATURES
    + LEXICAL_FEATURES
)


# =========================================================
# Data loading
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
            "split",
        }
    )

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{path} is missing columns: "
            f"{sorted(missing)}"
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


# =========================================================
# Model
# =========================================================

def build_model() -> Pipeline:
    """
    Frozen detector configuration selected before threshold
    optimization:

        retrieval + lexical features
        median imputation
        standard scaling
        balanced logistic regression
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
                    max_iter=2000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )


# =========================================================
# Threshold evaluation
# =========================================================

def evaluate_threshold(
    y_true: np.ndarray,
    sufficient_probability: np.ndarray,
    threshold: float,
) -> dict:
    """
    Original target:
        0 = insufficient
        1 = sufficient

    Operational positive class:
        insufficient evidence
    """

    y_pred = (
        sufficient_probability >= threshold
    ).astype(int)

    y_true_insufficient = (
        1 - y_true
    )

    y_pred_insufficient = (
        1 - y_pred
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
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

    true_sufficient = int(
        matrix[1, 0]
        + matrix[1, 1]
    )

    predicted_insufficient = int(
        (y_pred == 0).sum()
    )

    recovery_rate = (
        predicted_insufficient
        / len(y_pred)
    )

    false_recovery_rate = (
        false_recovery_flags
        / true_sufficient
        if true_sufficient > 0
        else 0.0
    )

    return {
        "threshold": float(
            threshold
        ),
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
        "detected_insufficient": (
            detected_insufficient
        ),
        "missed_insufficient": (
            missed_insufficient
        ),
        "false_recovery_flags": (
            false_recovery_flags
        ),
        "recovery_rate": (
            recovery_rate
        ),
        "false_recovery_rate": (
            false_recovery_rate
        ),
    }


# =========================================================
# Reporting
# =========================================================

def print_operating_point(
    title: str,
    row: pd.Series,
) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

    print(
        f"Threshold: "
        f"{row['threshold']:.2f}"
    )

    print(
        f"Accuracy: "
        f"{row['accuracy']:.4f}"
    )

    print(
        f"Balanced accuracy: "
        f"{row['balanced_accuracy']:.4f}"
    )

    print(
        f"Insufficient precision: "
        f"{row['insufficient_precision']:.4f}"
    )

    print(
        f"Insufficient recall: "
        f"{row['insufficient_recall']:.4f}"
    )

    print(
        f"Insufficient F1: "
        f"{row['insufficient_f1']:.4f}"
    )

    print(
        f"Detected insufficient: "
        f"{int(row['detected_insufficient'])}"
    )

    print(
        f"Missed insufficient: "
        f"{int(row['missed_insufficient'])}"
    )

    print(
        f"False recovery flags: "
        f"{int(row['false_recovery_flags'])}"
    )

    print(
        f"Recovery trigger rate: "
        f"{row['recovery_rate']:.4f}"
    )

    print(
        f"False recovery rate among "
        f"sufficient cases: "
        f"{row['false_recovery_rate']:.4f}"
    )


# =========================================================
# Main
# =========================================================

def main() -> None:
    print("=" * 90)

    print(
        "PHASE 9.7C — SUFFICIENCY "
        "OPERATING-THRESHOLD SELECTION"
    )

    print("=" * 90)

    print(
        "\nFrozen representation: "
        "retrieval + lexical"
    )

    print(
        f"Feature count: "
        f"{len(MODEL_FEATURES)}"
    )

    print(
        "Frozen model: "
        "class-weight-balanced logistic regression"
    )

    print(
        "Predeclared minimum insufficient "
        f"recall: {MIN_INSUFFICIENT_RECALL:.2f}"
    )

    print(
        "\nSelection rule:"
    )

    print(
        "1. Keep thresholds with insufficient "
        f"recall >= {MIN_INSUFFICIENT_RECALL:.2f}"
    )

    print(
        "2. Select highest insufficient precision."
    )

    print(
        "3. Break precision ties using higher "
        "balanced accuracy."
    )

    print(
        "4. Break remaining ties using higher "
        "threshold."
    )

    # ---------------------------------------------------------
    # Load frozen train / validation data
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

    print(
        f"Train rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    X_train = train_df[
        MODEL_FEATURES
    ].copy()

    y_train = train_df[
        TARGET_COLUMN
    ].astype(int).to_numpy()

    X_validation = validation_df[
        MODEL_FEATURES
    ].copy()

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
    # Train exactly once
    # ---------------------------------------------------------

    print(
        "\nTraining frozen detector..."
    )

    model = build_model()

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )

    validation_probability = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    # ---------------------------------------------------------
    # Threshold sweep
    # ---------------------------------------------------------

    rows = []

    for threshold in THRESHOLDS:
        rows.append(
            evaluate_threshold(
                y_true=y_validation,
                sufficient_probability=(
                    validation_probability
                ),
                threshold=float(
                    threshold
                ),
            )
        )

    results_df = pd.DataFrame(
        rows
    )

    qualifying_df = results_df[
        results_df[
            "insufficient_recall"
        ] >= MIN_INSUFFICIENT_RECALL
    ].copy()

    if qualifying_df.empty:
        raise RuntimeError(
            "No threshold satisfies the predeclared "
            "minimum insufficient recall."
        )

    # Predeclared selection rule.
    qualifying_df = (
        qualifying_df
        .sort_values(
            by=[
                "insufficient_precision",
                "balanced_accuracy",
                "threshold",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    selected = (
        qualifying_df.iloc[0]
    )

    # ---------------------------------------------------------
    # Reference threshold = 0.50
    # ---------------------------------------------------------

    reference_rows = results_df[
        np.isclose(
            results_df["threshold"],
            0.50,
        )
    ]

    if len(reference_rows) != 1:
        raise RuntimeError(
            "Could not uniquely identify "
            "threshold 0.50."
        )

    reference = (
        reference_rows.iloc[0]
    )

    print_operating_point(
        title=(
            "REFERENCE OPERATING POINT — "
            "THRESHOLD 0.50"
        ),
        row=reference,
    )

    print_operating_point(
        title=(
            "SELECTED VALIDATION "
            "OPERATING POINT"
        ),
        row=selected,
    )

    # ---------------------------------------------------------
    # Print local neighborhood around selected threshold
    # ---------------------------------------------------------

    selected_threshold = float(
        selected["threshold"]
    )

    neighborhood = results_df[
        (
            results_df["threshold"]
            >= selected_threshold - 0.05
        )
        & (
            results_df["threshold"]
            <= selected_threshold + 0.05
        )
    ].copy()

    display_columns = [
        "threshold",
        "accuracy",
        "balanced_accuracy",
        "insufficient_precision",
        "insufficient_recall",
        "insufficient_f1",
        "detected_insufficient",
        "missed_insufficient",
        "false_recovery_flags",
        "recovery_rate",
        "false_recovery_rate",
    ]

    neighborhood = neighborhood[
        display_columns
    ]

    float_columns = [
        "threshold",
        "accuracy",
        "balanced_accuracy",
        "insufficient_precision",
        "insufficient_recall",
        "insufficient_f1",
        "recovery_rate",
        "false_recovery_rate",
    ]

    neighborhood[
        float_columns
    ] = neighborhood[
        float_columns
    ].round(4)

    print("\n" + "=" * 120)

    print(
        "THRESHOLD NEIGHBORHOOD "
        "AROUND SELECTED POINT"
    )

    print("=" * 120)

    print()

    print(
        neighborhood.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Top qualifying thresholds
    # ---------------------------------------------------------

    top_qualifying = (
        qualifying_df[
            display_columns
        ]
        .head(10)
        .copy()
    )

    top_qualifying[
        float_columns
    ] = top_qualifying[
        float_columns
    ].round(4)

    print("\n" + "=" * 120)

    print(
        "TOP 10 QUALIFYING THRESHOLDS "
        "UNDER PREDECLARED RULE"
    )

    print("=" * 120)

    print()

    print(
        top_qualifying.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Freeze candidate configuration
    # ---------------------------------------------------------

    print("\n" + "=" * 90)

    print(
        "PHASE 9.7C COMPLETE"
    )

    print("=" * 90)

    print(
        "\nCandidate detector configuration:"
    )

    print(
        "Features: retrieval + lexical "
        f"({len(MODEL_FEATURES)})"
    )

    print(
        "Model: balanced logistic regression"
    )

    print(
        "Selected validation threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        "Threshold selection constraint: "
        f"insufficient recall >= "
        f"{MIN_INSUFFICIENT_RECALL:.2f}"
    )

    print(
        "\nIMPORTANT: This threshold was selected "
        "using validation data."
    )

    print(
        "No test examples were loaded or evaluated."
    )

    print(
        "The 500-question test split remains untouched."
    )


if __name__ == "__main__":
    main()