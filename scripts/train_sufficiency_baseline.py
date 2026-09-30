from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
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

# IMPORTANT:
# These are ONLY the runtime retrieval-statistics features
# produced by src/diagnosis/signals.py.
#
# No lexical evidence features.
# No semantic evidence features.
# No oracle/gold features.
# No question text.
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
# Loading
# =========================================================

def load_dataset(
    path: Path,
    expected_split: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    dataframe = pd.read_csv(path)

    if "split" not in dataframe.columns:
        raise RuntimeError(
            f"{path} does not contain a split column."
        )

    actual_splits = set(
        dataframe["split"]
        .astype(str)
        .unique()
    )

    if actual_splits != {expected_split}:
        raise RuntimeError(
            f"Expected split {expected_split!r}, "
            f"found {actual_splits}."
        )

    return dataframe


# =========================================================
# Schema / leakage checks
# =========================================================

def validate_columns(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> None:
    required_columns = (
        set(RETRIEVAL_FEATURES)
        | {TARGET_COLUMN}
    )

    for name, dataframe in (
        ("train", train_df),
        ("validation", validation_df),
    ):
        missing = (
            required_columns
            - set(dataframe.columns)
        )

        if missing:
            raise RuntimeError(
                f"{name} dataset is missing "
                f"required columns: {sorted(missing)}"
            )

    # These columns must NOT be part of the retrieval-only
    # baseline feature list.
    forbidden_model_features = {
        "question",
        "oracle_failure_type",
        "is_evidence_sufficient",

        # Lexical evidence features
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

        # Semantic evidence features
        "max_query_chunk_similarity",
        "mean_query_chunk_similarity",
        "semantic_coverage_spread",
        "mean_pairwise_chunk_similarity",
    }

    leaked = (
        set(RETRIEVAL_FEATURES)
        & forbidden_model_features
    )

    if leaked:
        raise RuntimeError(
            "Non-baseline features leaked into "
            f"retrieval baseline: {sorted(leaked)}"
        )


# =========================================================
# Metrics
# =========================================================

def evaluate(
    name: str,
    y_true: np.ndarray,
    sufficient_probability: np.ndarray,
    threshold: float = 0.5,
) -> None:
    """
    Labels:
        1 = sufficient evidence
        0 = insufficient evidence

    The model predicts probability of class 1
    (sufficient evidence).

    Operationally, however, insufficient evidence is the
    important condition because it can eventually trigger
    recovery.

    Therefore we report metrics with insufficient evidence
    treated as the positive class as well.
    """

    # ---------------------------------------------------------
    # Convert probabilities to predictions
    # ---------------------------------------------------------

    y_pred = (
        sufficient_probability >= threshold
    ).astype(int)

    # ---------------------------------------------------------
    # Standard metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    # ---------------------------------------------------------
    # Treat INSUFFICIENT as positive
    #
    # Original:
    #   0 = insufficient
    #   1 = sufficient
    #
    # Converted:
    #   1 = insufficient
    #   0 = sufficient
    # ---------------------------------------------------------

    y_true_insufficient = 1 - y_true
    y_pred_insufficient = 1 - y_pred

    insufficient_probability = (
        1.0 - sufficient_probability
    )

    insufficient_precision = precision_score(
        y_true_insufficient,
        y_pred_insufficient,
        zero_division=0,
    )

    insufficient_recall = recall_score(
        y_true_insufficient,
        y_pred_insufficient,
        zero_division=0,
    )

    insufficient_f1 = f1_score(
        y_true_insufficient,
        y_pred_insufficient,
        zero_division=0,
    )

    insufficient_roc_auc = roc_auc_score(
        y_true_insufficient,
        insufficient_probability,
    )

    insufficient_pr_auc = average_precision_score(
        y_true_insufficient,
        insufficient_probability,
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print(name)
    print("=" * 90)

    print(
        f"\nDecision threshold "
        f"(sufficient): {threshold:.2f}"
    )

    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy:.4f}"
    )

    print(
        "\nINSUFFICIENT-EVIDENCE detection"
    )

    print("-" * 90)

    print(
        f"Precision: "
        f"{insufficient_precision:.4f}"
    )

    print(
        f"Recall: "
        f"{insufficient_recall:.4f}"
    )

    print(
        f"F1: "
        f"{insufficient_f1:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{insufficient_roc_auc:.4f}"
    )

    print(
        f"PR-AUC: "
        f"{insufficient_pr_auc:.4f}"
    )

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    print(
        "\nConfusion matrix"
    )

    print(
        "Rows = true [insufficient, sufficient]\n"
        "Cols = predicted [insufficient, sufficient]"
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    print(matrix)

    # ---------------------------------------------------------
    # Full classification report
    # ---------------------------------------------------------

    print(
        "\nClassification report"
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[
            "insufficient",
            "sufficient",
        ],
        digits=4,
        zero_division=0,
    )

    print(report)


# =========================================================
# Main
# =========================================================

def main() -> None:
    print("=" * 90)
    print(
        "PHASE 9.6A — RETRIEVAL-STATISTICS "
        "SUFFICIENCY BASELINE"
    )
    print("=" * 90)

    # ---------------------------------------------------------
    # Load frozen train / validation data
    # ---------------------------------------------------------

    print(
        "\nLoading frozen datasets..."
    )

    train_df = load_dataset(
        TRAIN_PATH,
        expected_split="train",
    )

    validation_df = load_dataset(
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

    validate_columns(
        train_df,
        validation_df,
    )

    # ---------------------------------------------------------
    # Build X / y
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
        f"\nRetrieval features: "
        f"{len(RETRIEVAL_FEATURES)}"
    )

    print(
        "Train sufficient / insufficient: "
        f"{int(y_train.sum()):,} / "
        f"{int((1 - y_train).sum()):,}"
    )

    print(
        "Validation sufficient / insufficient: "
        f"{int(y_validation.sum()):,} / "
        f"{int((1 - y_validation).sum()):,}"
    )

    # ---------------------------------------------------------
    # Pipeline
    # ---------------------------------------------------------
    #
    # Median imputation is learned ONLY from training data.
    #
    # This handles undefined rank-disagreement values without
    # modifying the frozen source CSVs.
    #
    # Standardization is also learned ONLY from training data.
    # ---------------------------------------------------------

    numeric_pipeline = Pipeline(
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
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "retrieval",
                numeric_pipeline,
                RETRIEVAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    # Deliberately simple baseline.
    #
    # We do NOT use class_weight="balanced" yet.
    # We do NOT tune hyperparameters yet.
    # We do NOT tune the decision threshold yet.
    model = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    print(
        "\nTraining logistic-regression baseline..."
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )

    # ---------------------------------------------------------
    # Probabilities
    # ---------------------------------------------------------

    train_probability = (
        pipeline.predict_proba(
            X_train
        )[:, 1]
    )

    validation_probability = (
        pipeline.predict_proba(
            X_validation
        )[:, 1]
    )

    # ---------------------------------------------------------
    # Majority baseline
    # ---------------------------------------------------------

    majority_prediction = np.ones_like(
        y_validation
    )

    majority_accuracy = accuracy_score(
        y_validation,
        majority_prediction,
    )

    majority_balanced_accuracy = (
        balanced_accuracy_score(
            y_validation,
            majority_prediction,
        )
    )

    majority_insufficient_recall = recall_score(
        1 - y_validation,
        1 - majority_prediction,
        zero_division=0,
    )

    print("\n" + "=" * 90)
    print(
        "TRIVIAL MAJORITY BASELINE"
    )
    print("=" * 90)

    print(
        "\nAlways predicts: sufficient"
    )

    print(
        f"Validation accuracy: "
        f"{majority_accuracy:.4f}"
    )

    print(
        f"Validation balanced accuracy: "
        f"{majority_balanced_accuracy:.4f}"
    )

    print(
        f"Insufficient recall: "
        f"{majority_insufficient_recall:.4f}"
    )

    # ---------------------------------------------------------
    # Evaluate fixed 0.5 threshold
    # ---------------------------------------------------------

    evaluate(
        name=(
            "TRAIN — RETRIEVAL STATS ONLY"
        ),
        y_true=y_train,
        sufficient_probability=(
            train_probability
        ),
        threshold=0.5,
    )

    evaluate(
        name=(
            "VALIDATION — RETRIEVAL STATS ONLY"
        ),
        y_true=y_validation,
        sufficient_probability=(
            validation_probability
        ),
        threshold=0.5,
    )

    # ---------------------------------------------------------
    # Completion
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print(
        "PHASE 9.6A COMPLETE"
    )
    print("=" * 90)

    print(
        "\nNOTE: This is intentionally a simple "
        "retrieval-statistics-only baseline."
    )

    print(
        "No lexical, semantic, oracle, gold, "
        "or question-text features were used."
    )

    print(
        "Missing rank-disagreement values were "
        "median-imputed using training data only."
    )

    print(
        "The decision threshold remains fixed at 0.5."
    )

    print(
        "The 500-question test split remains untouched."
    )


if __name__ == "__main__":
    main()