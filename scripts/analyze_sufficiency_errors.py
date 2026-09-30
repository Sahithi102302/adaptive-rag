from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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

FROZEN_THRESHOLD = 0.58


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


# A smaller interpretable set for aggregate error analysis.
ANALYSIS_FEATURES = [
    "top_score",
    "score_margin",
    "unique_document_count",
    "max_document_fraction",
    "both_retriever_fraction",
    "dense_only_fraction",
    "informative_query_token_count",
    "query_token_coverage",
    "uncovered_query_token_count",
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
        set(MODEL_FEATURES)
        | {
            TARGET_COLUMN,
            ORACLE_COLUMN,
            "split",
            "question_index",
            "question",
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
# Prediction categories
# =========================================================

def assign_outcome(
    true_label: int,
    predicted_label: int,
) -> str:
    """
    Target:
        0 = insufficient
        1 = sufficient

    Operational interpretation:

        true 0, pred 0 -> detected_failure
        true 0, pred 1 -> missed_failure
        true 1, pred 0 -> false_recovery
        true 1, pred 1 -> correct_sufficient
    """

    if true_label == 0 and predicted_label == 0:
        return "detected_failure"

    if true_label == 0 and predicted_label == 1:
        return "missed_failure"

    if true_label == 1 and predicted_label == 0:
        return "false_recovery"

    if true_label == 1 and predicted_label == 1:
        return "correct_sufficient"

    raise RuntimeError(
        "Unexpected binary labels."
    )


# =========================================================
# Reporting helpers
# =========================================================

def print_oracle_breakdown(
    validation_df: pd.DataFrame,
) -> None:
    print("\n" + "=" * 100)
    print(
        "FAILURE DETECTION BY ORACLE FAILURE TYPE"
    )
    print("=" * 100)

    failures = validation_df[
        validation_df[
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
                group["prediction_outcome"]
                == "detected_failure"
            ).sum()
        )

        missed = int(
            (
                group["prediction_outcome"]
                == "missed_failure"
            ).sum()
        )

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

    result = pd.DataFrame(
        rows
    )

    result[
        "detection_recall"
    ] = result[
        "detection_recall"
    ].round(4)

    print()
    print(
        result.to_string(
            index=False
        )
    )


def print_outcome_feature_means(
    validation_df: pd.DataFrame,
) -> None:
    print("\n" + "=" * 120)
    print(
        "MEAN RUNTIME FEATURES BY "
        "PREDICTION OUTCOME"
    )
    print("=" * 120)

    outcome_order = [
        "detected_failure",
        "missed_failure",
        "false_recovery",
        "correct_sufficient",
    ]

    available_outcomes = [
        outcome
        for outcome in outcome_order
        if outcome
        in set(
            validation_df[
                "prediction_outcome"
            ]
        )
    ]

    means = (
        validation_df
        .groupby(
            "prediction_outcome"
        )[
            ANALYSIS_FEATURES
        ]
        .mean()
        .reindex(
            available_outcomes
        )
        .round(4)
    )

    counts = (
        validation_df[
            "prediction_outcome"
        ]
        .value_counts()
        .reindex(
            available_outcomes
        )
    )

    means.insert(
        0,
        "count",
        counts,
    )

    print()
    print(
        means.to_string()
    )


def print_probability_summary(
    validation_df: pd.DataFrame,
) -> None:
    print("\n" + "=" * 110)
    print(
        "P(SUFFICIENT) DISTRIBUTION "
        "BY OUTCOME"
    )
    print("=" * 110)

    rows = []

    for outcome, group in (
        validation_df.groupby(
            "prediction_outcome",
            sort=True,
        )
    ):
        values = (
            group[
                "sufficient_probability"
            ]
            .to_numpy(
                dtype=float
            )
        )

        rows.append(
            {
                "outcome": outcome,
                "count": len(values),
                "mean": np.mean(values),
                "min": np.min(values),
                "q25": np.quantile(
                    values,
                    0.25,
                ),
                "median": np.median(
                    values
                ),
                "q75": np.quantile(
                    values,
                    0.75,
                ),
                "max": np.max(values),
            }
        )

    result = pd.DataFrame(
        rows
    )

    numeric = [
        "mean",
        "min",
        "q25",
        "median",
        "q75",
        "max",
    ]

    result[
        numeric
    ] = result[
        numeric
    ].round(4)

    print()
    print(
        result.to_string(
            index=False
        )
    )


def print_missed_failures(
    validation_df: pd.DataFrame,
) -> None:
    print("\n" + "=" * 120)
    print(
        "ALL MISSED FAILURES "
        "(FALSE NEGATIVES)"
    )
    print("=" * 120)

    missed = validation_df[
        validation_df[
            "prediction_outcome"
        ] == "missed_failure"
    ].copy()

    missed = missed.sort_values(
        by="sufficient_probability",
        ascending=False,
    )

    columns = [
        "question_index",
        "oracle_failure_type",
        "sufficient_probability",
        "query_token_coverage",
        "uncovered_query_token_count",
        "top_score",
        "score_margin",
        "unique_document_count",
        "question",
    ]

    display = missed[
        columns
    ].copy()

    float_columns = [
        "sufficient_probability",
        "query_token_coverage",
        "top_score",
        "score_margin",
    ]

    display[
        float_columns
    ] = display[
        float_columns
    ].round(4)

    print(
        f"\nMissed failures: "
        f"{len(display)}"
    )

    print()

    with pd.option_context(
        "display.max_colwidth",
        100,
        "display.width",
        220,
    ):
        print(
            display.to_string(
                index=False
            )
        )


def print_false_recoveries(
    validation_df: pd.DataFrame,
) -> None:
    print("\n" + "=" * 120)
    print(
        "FALSE RECOVERY ANALYSIS"
    )
    print("=" * 120)

    false_recovery = validation_df[
        validation_df[
            "prediction_outcome"
        ] == "false_recovery"
    ].copy()

    print(
        f"\nFalse recovery cases: "
        f"{len(false_recovery)}"
    )

    if false_recovery.empty:
        return

    summary = (
        false_recovery[
            ANALYSIS_FEATURES
        ]
        .mean()
        .round(4)
    )

    print(
        "\nMean runtime features:"
    )

    print(
        summary.to_string()
    )

    # Show the most confident false recovery cases:
    # lowest P(sufficient) even though oracle says sufficient.
    examples = (
        false_recovery
        .sort_values(
            by="sufficient_probability",
            ascending=True,
        )
        .head(15)
        .copy()
    )

    columns = [
        "question_index",
        "sufficient_probability",
        "query_token_coverage",
        "uncovered_query_token_count",
        "top_score",
        "score_margin",
        "unique_document_count",
        "question",
    ]

    display = examples[
        columns
    ].copy()

    float_columns = [
        "sufficient_probability",
        "query_token_coverage",
        "top_score",
        "score_margin",
    ]

    display[
        float_columns
    ] = display[
        float_columns
    ].round(4)

    print(
        "\n15 most confident false-recovery cases:"
    )

    print()

    with pd.option_context(
        "display.max_colwidth",
        100,
        "display.width",
        220,
    ):
        print(
            display.to_string(
                index=False
            )
        )


# =========================================================
# Main
# =========================================================

def main() -> None:
    print("=" * 100)

    print(
        "PHASE 9.7D — FROZEN SUFFICIENCY "
        "DETECTOR ERROR ANALYSIS"
    )

    print("=" * 100)

    print(
        "\nFrozen detector:"
    )

    print(
        f"  Features: retrieval + lexical "
        f"({len(MODEL_FEATURES)})"
    )

    print(
        "  Model: balanced logistic regression"
    )

    print(
        f"  Threshold: "
        f"{FROZEN_THRESHOLD:.2f}"
    )

    # ---------------------------------------------------------
    # Load frozen data
    # ---------------------------------------------------------

    print(
        "\nLoading frozen train and "
        "validation datasets..."
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

    # ---------------------------------------------------------
    # Train frozen model
    # ---------------------------------------------------------

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
        "\nTraining frozen detector..."
    )

    model = build_frozen_model()

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )

    # ---------------------------------------------------------
    # Frozen predictions
    # ---------------------------------------------------------

    sufficient_probability = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    predicted_label = (
        sufficient_probability
        >= FROZEN_THRESHOLD
    ).astype(int)

    validation_df = (
        validation_df.copy()
    )

    validation_df[
        "sufficient_probability"
    ] = sufficient_probability

    validation_df[
        "predicted_sufficient"
    ] = predicted_label

    validation_df[
        "prediction_outcome"
    ] = [
        assign_outcome(
            int(true_label),
            int(pred_label),
        )
        for true_label, pred_label
        in zip(
            y_validation,
            predicted_label,
        )
    ]

    # ---------------------------------------------------------
    # Sanity checks
    # ---------------------------------------------------------

    outcome_counts = (
        validation_df[
            "prediction_outcome"
        ]
        .value_counts()
    )

    detected = int(
        outcome_counts.get(
            "detected_failure",
            0,
        )
    )

    missed = int(
        outcome_counts.get(
            "missed_failure",
            0,
        )
    )

    false_recovery = int(
        outcome_counts.get(
            "false_recovery",
            0,
        )
    )

    correct_sufficient = int(
        outcome_counts.get(
            "correct_sufficient",
            0,
        )
    )

    print("\n" + "=" * 100)

    print(
        "FROZEN VALIDATION OUTCOMES"
    )

    print("=" * 100)

    print(
        f"\nDetected failures: "
        f"{detected}"
    )

    print(
        f"Missed failures: "
        f"{missed}"
    )

    print(
        f"False recovery flags: "
        f"{false_recovery}"
    )

    print(
        f"Correct sufficient: "
        f"{correct_sufficient}"
    )

    if (
        detected != 45
        or missed != 18
        or false_recovery != 45
        or correct_sufficient != 92
    ):
        raise RuntimeError(
            "Frozen predictions do not reproduce "
            "Phase 9.7C threshold results."
        )

    print(
        "\nPASS: Phase 9.7C operating point "
        "reproduced exactly."
    )

    # ---------------------------------------------------------
    # Error analyses
    # ---------------------------------------------------------

    print_oracle_breakdown(
        validation_df
    )

    print_outcome_feature_means(
        validation_df
    )

    print_probability_summary(
        validation_df
    )

    print_missed_failures(
        validation_df
    )

    print_false_recoveries(
        validation_df
    )

    # ---------------------------------------------------------
    # Completion
    # ---------------------------------------------------------

    print("\n" + "=" * 100)

    print(
        "PHASE 9.7D COMPLETE"
    )

    print("=" * 100)

    print(
        "\nThis was offline error analysis only."
    )

    print(
        "Oracle failure labels were used only "
        "after prediction for analysis."
    )

    print(
        "No oracle labels were model inputs."
    )

    print(
        "No feature, model, class-weight, or "
        "threshold changes were made."
    )

    print(
        "No test examples were loaded or evaluated."
    )

    print(
        "The 500-question test split remains untouched."
    )


if __name__ == "__main__":
    main()