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
        set(RETRIEVAL_FEATURES)
        | {
            TARGET_COLUMN,
            "split",
        }
    )

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing columns in {path}: "
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

def build_model(
    class_weight,
) -> Pipeline:
    """
    Both models use exactly the same preprocessing.

    Only class_weight changes.

    This isolates the effect of class balancing.
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
                    class_weight=class_weight,
                ),
            ),
        ]
    )


# =========================================================
# Metrics
# =========================================================

def calculate_metrics(
    y_true: np.ndarray,
    sufficient_probability: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Original target:
        0 = insufficient
        1 = sufficient

    For failure detection metrics we convert:
        1 = insufficient
        0 = sufficient
    """

    y_pred = (
        sufficient_probability >= threshold
    ).astype(int)

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
        "confusion_matrix": matrix,
    }


def print_metrics(
    name: str,
    metrics: dict,
) -> None:
    print("\n" + "=" * 90)
    print(name)
    print("=" * 90)

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        "\nINSUFFICIENT-EVIDENCE detection"
    )

    print(
        f"Precision: "
        f"{metrics['insufficient_precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{metrics['insufficient_recall']:.4f}"
    )

    print(
        f"F1: "
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
        "\nConfusion matrix"
    )

    print(
        "Rows = true [insufficient, sufficient]\n"
        "Cols = predicted [insufficient, sufficient]"
    )

    print(
        metrics["confusion_matrix"]
    )


# =========================================================
# Coefficients
# =========================================================

def print_coefficients(
    pipeline: Pipeline,
    title: str,
) -> None:
    """
    Coefficients operate on standardized features.

    Positive coefficient:
        pushes prediction toward SUFFICIENT.

    Negative coefficient:
        pushes prediction toward INSUFFICIENT.

    Because preprocessing is identical and features are
    standardized, coefficient magnitudes are more directly
    comparable within a model.
    """

    model = pipeline.named_steps["model"]

    coefficients = model.coef_[0]

    coefficient_df = pd.DataFrame(
        {
            "feature": RETRIEVAL_FEATURES,
            "coefficient": coefficients,
        }
    )

    coefficient_df[
        "absolute_coefficient"
    ] = (
        coefficient_df["coefficient"]
        .abs()
    )

    coefficient_df = (
        coefficient_df
        .sort_values(
            "absolute_coefficient",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

    print(
        "\nPositive -> predicts more SUFFICIENT"
    )

    print(
        "Negative -> predicts more INSUFFICIENT\n"
    )

    for _, row in coefficient_df.iterrows():
        feature = row["feature"]
        coefficient = row["coefficient"]

        print(
            f"{feature:<35} "
            f"{coefficient:>10.4f}"
        )


# =========================================================
# Main
# =========================================================

def main() -> None:
    print("=" * 90)
    print(
        "PHASE 9.6B — RETRIEVAL-ONLY "
        "SUFFICIENCY BASELINE COMPARISON"
    )
    print("=" * 90)

    # ---------------------------------------------------------
    # Load frozen data
    # ---------------------------------------------------------

    print(
        "\nLoading frozen train and validation datasets..."
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

    print(
        f"Features: "
        f"{len(RETRIEVAL_FEATURES)}"
    )

    # ---------------------------------------------------------
    # X / y
    # ---------------------------------------------------------

    X_train = train_df[
        RETRIEVAL_FEATURES
    ].copy()

    y_train = train_df[
        TARGET_COLUMN
    ].astype(int).to_numpy()

    X_validation = validation_df[
        RETRIEVAL_FEATURES
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
    # Model A: plain logistic regression
    # ---------------------------------------------------------

    print(
        "\nTraining plain logistic regression..."
    )

    plain_model = build_model(
        class_weight=None,
    )

    plain_model.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # Model B: class-balanced logistic regression
    # ---------------------------------------------------------

    print(
        "Training class-balanced logistic regression..."
    )

    balanced_model = build_model(
        class_weight="balanced",
    )

    balanced_model.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )

    # ---------------------------------------------------------
    # Validation probabilities
    # ---------------------------------------------------------

    plain_probability = (
        plain_model.predict_proba(
            X_validation
        )[:, 1]
    )

    balanced_probability = (
        balanced_model.predict_proba(
            X_validation
        )[:, 1]
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    plain_metrics = calculate_metrics(
        y_true=y_validation,
        sufficient_probability=(
            plain_probability
        ),
        threshold=0.5,
    )

    balanced_metrics = calculate_metrics(
        y_true=y_validation,
        sufficient_probability=(
            balanced_probability
        ),
        threshold=0.5,
    )

    print_metrics(
        name=(
            "VALIDATION — PLAIN LOGISTIC REGRESSION"
        ),
        metrics=plain_metrics,
    )

    print_metrics(
        name=(
            "VALIDATION — CLASS-WEIGHTED "
            "LOGISTIC REGRESSION"
        ),
        metrics=balanced_metrics,
    )

    # ---------------------------------------------------------
    # Direct comparison
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print("DIRECT VALIDATION COMPARISON")
    print("=" * 90)

    comparison = pd.DataFrame(
        {
            "plain": {
                "accuracy": (
                    plain_metrics["accuracy"]
                ),
                "balanced_accuracy": (
                    plain_metrics[
                        "balanced_accuracy"
                    ]
                ),
                "insufficient_precision": (
                    plain_metrics[
                        "insufficient_precision"
                    ]
                ),
                "insufficient_recall": (
                    plain_metrics[
                        "insufficient_recall"
                    ]
                ),
                "insufficient_f1": (
                    plain_metrics[
                        "insufficient_f1"
                    ]
                ),
                "roc_auc": (
                    plain_metrics["roc_auc"]
                ),
                "pr_auc": (
                    plain_metrics["pr_auc"]
                ),
            },
            "class_weight_balanced": {
                "accuracy": (
                    balanced_metrics["accuracy"]
                ),
                "balanced_accuracy": (
                    balanced_metrics[
                        "balanced_accuracy"
                    ]
                ),
                "insufficient_precision": (
                    balanced_metrics[
                        "insufficient_precision"
                    ]
                ),
                "insufficient_recall": (
                    balanced_metrics[
                        "insufficient_recall"
                    ]
                ),
                "insufficient_f1": (
                    balanced_metrics[
                        "insufficient_f1"
                    ]
                ),
                "roc_auc": (
                    balanced_metrics["roc_auc"]
                ),
                "pr_auc": (
                    balanced_metrics["pr_auc"]
                ),
            },
        }
    )

    print()
    print(
        comparison.round(4).to_string()
    )

    # ---------------------------------------------------------
    # Coefficients
    # ---------------------------------------------------------

    print_coefficients(
        pipeline=plain_model,
        title=(
            "STANDARDIZED COEFFICIENTS — "
            "PLAIN MODEL"
        ),
    )

    print_coefficients(
        pipeline=balanced_model,
        title=(
            "STANDARDIZED COEFFICIENTS — "
            "CLASS-WEIGHTED MODEL"
        ),
    )

    # ---------------------------------------------------------
    # Completion
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print("PHASE 9.6B COMPLETE")
    print("=" * 90)

    print(
        "\nBoth models used exactly the same "
        "19 retrieval-statistics features."
    )

    print(
        "The only modeling difference was "
        "class_weight=None versus "
        "class_weight='balanced'."
    )

    print(
        "Both models used a fixed 0.5 threshold."
    )

    print(
        "No lexical, semantic, question-text, "
        "oracle, or gold features were used."
    )

    print(
        "The test split remains untouched."
    )


if __name__ == "__main__":
    main()