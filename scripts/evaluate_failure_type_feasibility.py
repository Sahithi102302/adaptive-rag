from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
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
# Configuration
# =========================================================

ARTIFACT_DIR = Path("artifacts/diagnosis/scaled")

TRAIN_PATH = ARTIFACT_DIR / "diagnostic_train.csv"
VALIDATION_PATH = ARTIFACT_DIR / "diagnostic_validation.csv"

TARGET_COLUMN = "is_evidence_sufficient"
ORACLE_COLUMN = "oracle_failure_type"

COVERAGE = "evidence_coverage_failure"
PASSAGE = "passage_selection_failure"

# Binary target used ONLY inside this feasibility experiment:
#   0 = evidence coverage failure
#   1 = passage selection failure
PASSAGE_TARGET = "is_passage_selection_failure"


# =========================================================
# Existing runtime feature families
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


FEATURE_FAMILIES = {
    "retrieval_only": RETRIEVAL_FEATURES,

    "retrieval_plus_lexical": (
        RETRIEVAL_FEATURES
        + LEXICAL_FEATURES
    ),

    "retrieval_plus_semantic": (
        RETRIEVAL_FEATURES
        + SEMANTIC_FEATURES
    ),

    "all_runtime_features": (
        RETRIEVAL_FEATURES
        + LEXICAL_FEATURES
        + SEMANTIC_FEATURES
    ),
}


# =========================================================
# Loading
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

    all_features = set()

    for features in FEATURE_FAMILIES.values():
        all_features.update(features)

    required = (
        all_features
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


def prepare_binary_failure_subset(
    df: pd.DataFrame,
) -> pd.DataFrame:

    # Only oracle-insufficient examples.
    failures = df[
        df[TARGET_COLUMN] == 0
    ].copy()

    # Controlled comparison:
    # coverage vs passage only.
    failures = failures[
        failures[ORACLE_COLUMN].isin(
            [COVERAGE, PASSAGE]
        )
    ].copy()

    failures[PASSAGE_TARGET] = (
        failures[ORACLE_COLUMN]
        == PASSAGE
    ).astype(int)

    return failures


# =========================================================
# Model
# =========================================================

def build_model() -> Pipeline:

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


# =========================================================
# Majority baseline
# =========================================================

def evaluate_majority_baseline(
    y_validation: np.ndarray,
) -> dict:

    # Always predict coverage:
    # passage target = 0.
    predictions = np.zeros_like(
        y_validation
    )

    return {
        "model": "majority_coverage_baseline",
        "feature_count": 0,
        "accuracy": accuracy_score(
            y_validation,
            predictions,
        ),
        "balanced_accuracy": (
            balanced_accuracy_score(
                y_validation,
                predictions,
            )
        ),
        "passage_precision": (
            precision_score(
                y_validation,
                predictions,
                zero_division=0,
            )
        ),
        "passage_recall": (
            recall_score(
                y_validation,
                predictions,
                zero_division=0,
            )
        ),
        "passage_f1": (
            f1_score(
                y_validation,
                predictions,
                zero_division=0,
            )
        ),
        "macro_f1": (
            f1_score(
                y_validation,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
    }


# =========================================================
# Feature-family experiment
# =========================================================

def evaluate_feature_family(
    name: str,
    features: list[str],
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> tuple[dict, np.ndarray, np.ndarray]:

    X_train = train_df[
        features
    ].copy()

    y_train = train_df[
        PASSAGE_TARGET
    ].astype(int).to_numpy()

    X_validation = validation_df[
        features
    ].copy()

    y_validation = validation_df[
        PASSAGE_TARGET
    ].astype(int).to_numpy()

    model = build_model()

    model.fit(
        X_train,
        y_train,
    )

    # Fixed default probability threshold.
    # No threshold tuning in this feasibility experiment.
    passage_probability = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    predictions = (
        passage_probability >= 0.50
    ).astype(int)

    result = {
        "model": name,
        "feature_count": len(features),

        "accuracy": accuracy_score(
            y_validation,
            predictions,
        ),

        "balanced_accuracy": (
            balanced_accuracy_score(
                y_validation,
                predictions,
            )
        ),

        "passage_precision": (
            precision_score(
                y_validation,
                predictions,
                zero_division=0,
            )
        ),

        "passage_recall": (
            recall_score(
                y_validation,
                predictions,
                zero_division=0,
            )
        ),

        "passage_f1": (
            f1_score(
                y_validation,
                predictions,
                zero_division=0,
            )
        ),

        "macro_f1": (
            f1_score(
                y_validation,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),

        "roc_auc": (
            roc_auc_score(
                y_validation,
                passage_probability,
            )
        ),

        "pr_auc": (
            average_precision_score(
                y_validation,
                passage_probability,
            )
        ),
    }

    return (
        result,
        predictions,
        passage_probability,
    )


# =========================================================
# Main
# =========================================================

def main() -> None:

    print("=" * 110)
    print(
        "PHASE 9.8B — CONTROLLED "
        "COVERAGE-VS-PASSAGE DIAGNOSIS "
        "FEASIBILITY"
    )
    print("=" * 110)

    print(
        "\nResearch question:"
    )

    print(
        "Among oracle-insufficient cases, "
        "can existing runtime signals "
        "distinguish evidence-coverage "
        "failures from passage-selection "
        "failures?"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This is a feasibility experiment, "
        "not the final runtime diagnoser."
    )

    print(
        "The held-out test split is not loaded."
    )

    print(
        "No threshold search is performed."
    )

    # -----------------------------------------------------
    # Load development data
    # -----------------------------------------------------

    print(
        "\nLoading train and validation..."
    )

    train_full = load_split(
        TRAIN_PATH,
        expected_split="train",
    )

    validation_full = load_split(
        VALIDATION_PATH,
        expected_split="validation",
    )

    train_df = (
        prepare_binary_failure_subset(
            train_full
        )
    )

    validation_df = (
        prepare_binary_failure_subset(
            validation_full
        )
    )

    print(
        f"\nTrain comparison rows: "
        f"{len(train_df)}"
    )

    print(
        f"  Coverage: "
        f"{int((train_df[PASSAGE_TARGET] == 0).sum())}"
    )

    print(
        f"  Passage: "
        f"{int((train_df[PASSAGE_TARGET] == 1).sum())}"
    )

    print(
        f"\nValidation comparison rows: "
        f"{len(validation_df)}"
    )

    print(
        f"  Coverage: "
        f"{int((validation_df[PASSAGE_TARGET] == 0).sum())}"
    )

    print(
        f"  Passage: "
        f"{int((validation_df[PASSAGE_TARGET] == 1).sum())}"
    )

    # Structural assertions.
    if len(train_df) != 286:
        raise RuntimeError(
            "Expected 286 train comparison rows."
        )

    if len(validation_df) != 62:
        raise RuntimeError(
            "Expected 62 validation comparison rows."
        )

    if int(
        train_df[PASSAGE_TARGET].sum()
    ) != 16:
        raise RuntimeError(
            "Expected 16 train passage failures."
        )

    if int(
        validation_df[PASSAGE_TARGET].sum()
    ) != 9:
        raise RuntimeError(
            "Expected 9 validation passage failures."
        )

    y_validation = validation_df[
        PASSAGE_TARGET
    ].astype(int).to_numpy()

    # -----------------------------------------------------
    # Majority baseline
    # -----------------------------------------------------

    results = [
        evaluate_majority_baseline(
            y_validation
        )
    ]

    detailed_outputs = {}

    # -----------------------------------------------------
    # Controlled feature-family ablation
    # -----------------------------------------------------

    for name, features in (
        FEATURE_FAMILIES.items()
    ):

        print(
            f"\nTraining: {name} "
            f"({len(features)} features)"
        )

        (
            result,
            predictions,
            probabilities,
        ) = evaluate_feature_family(
            name=name,
            features=features,
            train_df=train_df,
            validation_df=validation_df,
        )

        results.append(
            result
        )

        detailed_outputs[name] = {
            "predictions": predictions,
            "probabilities": probabilities,
        }

    # -----------------------------------------------------
    # Comparison table
    # -----------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    numeric_columns = [
        "accuracy",
        "balanced_accuracy",
        "passage_precision",
        "passage_recall",
        "passage_f1",
        "macro_f1",
        "roc_auc",
        "pr_auc",
    ]

    results_df[
        numeric_columns
    ] = results_df[
        numeric_columns
    ].round(4)

    print("\n" + "=" * 140)

    print(
        "VALIDATION FEASIBILITY RESULTS"
    )

    print("=" * 140)

    print()

    print(
        results_df.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Confusion matrices
    # -----------------------------------------------------

    print("\n" + "=" * 110)

    print(
        "CONFUSION MATRICES"
    )

    print("=" * 110)

    print(
        "\nRows = true "
        "[coverage, passage]"
    )

    print(
        "Cols = predicted "
        "[coverage, passage]"
    )

    baseline_predictions = (
        np.zeros_like(
            y_validation
        )
    )

    baseline_matrix = (
        confusion_matrix(
            y_validation,
            baseline_predictions,
            labels=[0, 1],
        )
    )

    print(
        "\nmajority_coverage_baseline"
    )

    print(
        baseline_matrix
    )

    for name, output in (
        detailed_outputs.items()
    ):

        matrix = confusion_matrix(
            y_validation,
            output["predictions"],
            labels=[0, 1],
        )

        print(
            f"\n{name}"
        )

        print(
            matrix
        )

    # -----------------------------------------------------
    # Minority-class probability inspection
    # -----------------------------------------------------

    print("\n" + "=" * 110)

    print(
        "PASSAGE-SELECTION PROBABILITY "
        "SUMMARY"
    )

    print("=" * 110)

    for name, output in (
        detailed_outputs.items()
    ):

        probabilities = (
            output["probabilities"]
        )

        coverage_probs = probabilities[
            y_validation == 0
        ]

        passage_probs = probabilities[
            y_validation == 1
        ]

        print(
            f"\n{name}"
        )

        print(
            "  Coverage examples:"
            f" mean={coverage_probs.mean():.4f},"
            f" median={np.median(coverage_probs):.4f},"
            f" min={coverage_probs.min():.4f},"
            f" max={coverage_probs.max():.4f}"
        )

        print(
            "  Passage examples:"
            f" mean={passage_probs.mean():.4f},"
            f" median={np.median(passage_probs):.4f},"
            f" min={passage_probs.min():.4f},"
            f" max={passage_probs.max():.4f}"
        )

    # -----------------------------------------------------
    # Per-example passage cases
    # -----------------------------------------------------

    print("\n" + "=" * 110)

    print(
        "VALIDATION PASSAGE-SELECTION CASES"
    )

    print("=" * 110)

    passage_rows = (
        validation_df[
            validation_df[
                PASSAGE_TARGET
            ] == 1
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    passage_positions = np.where(
        y_validation == 1
    )[0]

    for name, output in (
        detailed_outputs.items()
    ):

        passage_rows[
            f"{name}_prob"
        ] = output[
            "probabilities"
        ][
            passage_positions
        ]

        passage_rows[
            f"{name}_pred"
        ] = output[
            "predictions"
        ][
            passage_positions
        ]

    display_columns = [
        "question_index",
        "question",
    ]

    for name in FEATURE_FAMILIES:
        display_columns.extend(
            [
                f"{name}_prob",
                f"{name}_pred",
            ]
        )

    print()

    print(
        passage_rows[
            display_columns
        ].to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Completion
    # -----------------------------------------------------

    print("\n" + "=" * 110)

    print(
        "PHASE 9.8B COMPLETE"
    )

    print("=" * 110)

    print(
        "\nInterpretation constraints:"
    )

    print(
        "1. Passage-selection has only "
        "16 train and 9 validation examples."
    )

    print(
        "2. Accuracy is misleading because "
        "coverage failures dominate."
    )

    print(
        "3. Balanced accuracy, macro-F1, "
        "passage recall, ROC-AUC, and PR-AUC "
        "are more informative."
    )

    print(
        "4. Validation is used only for "
        "feasibility comparison; no threshold "
        "was tuned."
    )

    print(
        "5. Severe-retrieval failures were "
        "excluded because class support "
        "is inadequate."
    )

    print(
        "6. The held-out test split was "
        "not loaded or inspected."
    )


if __name__ == "__main__":
    main()