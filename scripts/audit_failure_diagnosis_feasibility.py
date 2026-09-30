from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# Configuration
# =========================================================

ARTIFACT_DIR = Path("artifacts/diagnosis/scaled")

TRAIN_PATH = ARTIFACT_DIR / "diagnostic_train.csv"
VALIDATION_PATH = ARTIFACT_DIR / "diagnostic_validation.csv"

TARGET_COLUMN = "is_evidence_sufficient"
ORACLE_COLUMN = "oracle_failure_type"


# =========================================================
# Runtime feature families
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


ALL_RUNTIME_FEATURES = (
    RETRIEVAL_FEATURES
    + LEXICAL_FEATURES
    + SEMANTIC_FEATURES
)


# =========================================================
# Helpers
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
        set(ALL_RUNTIME_FEATURES)
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

    actual = set(
        df["split"].astype(str).unique()
    )

    if actual != {expected_split}:
        raise RuntimeError(
            f"Expected split {expected_split!r}; "
            f"found {actual}."
        )

    return df


def failure_only(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df[
        df[TARGET_COLUMN] == 0
    ].copy()

    if (
        result[ORACLE_COLUMN]
        == "sufficient_evidence"
    ).any():
        raise RuntimeError(
            "Failure subset contains "
            "sufficient_evidence labels."
        )

    return result


def print_distribution(
    name: str,
    df: pd.DataFrame,
) -> None:

    print("\n" + "=" * 90)
    print(f"{name} FAILURE DISTRIBUTION")
    print("=" * 90)

    counts = (
        df[ORACLE_COLUMN]
        .value_counts()
        .rename_axis(
            "oracle_failure_type"
        )
        .reset_index(
            name="count"
        )
    )

    counts["fraction"] = (
        counts["count"] / len(df)
    ).round(4)

    print()
    print(
        counts.to_string(
            index=False
        )
    )


def standardized_mean_difference(
    x_a: pd.Series,
    x_b: pd.Series,
) -> float | None:
    """
    Absolute standardized mean difference:

        |mean_a - mean_b| / pooled_std

    Used only as a descriptive separation statistic.
    It is NOT a classifier-performance metric.
    """

    a = pd.to_numeric(
        x_a,
        errors="coerce",
    ).dropna().to_numpy(
        dtype=float
    )

    b = pd.to_numeric(
        x_b,
        errors="coerce",
    ).dropna().to_numpy(
        dtype=float
    )

    if len(a) < 2 or len(b) < 2:
        return None

    var_a = np.var(
        a,
        ddof=1,
    )

    var_b = np.var(
        b,
        ddof=1,
    )

    pooled_variance = (
        (
            (len(a) - 1) * var_a
            + (len(b) - 1) * var_b
        )
        / (
            len(a)
            + len(b)
            - 2
        )
    )

    if pooled_variance <= 0:
        return 0.0

    pooled_std = np.sqrt(
        pooled_variance
    )

    return float(
        abs(
            np.mean(a)
            - np.mean(b)
        )
        / pooled_std
    )


# =========================================================
# Coverage vs passage audit
# =========================================================

def audit_coverage_vs_passage(
    name: str,
    failures: pd.DataFrame,
) -> None:

    coverage = failures[
        failures[ORACLE_COLUMN]
        == "evidence_coverage_failure"
    ].copy()

    passage = failures[
        failures[ORACLE_COLUMN]
        == "passage_selection_failure"
    ].copy()

    print("\n" + "=" * 120)
    print(
        f"{name}: EVIDENCE COVERAGE "
        "VS PASSAGE SELECTION"
    )
    print("=" * 120)

    print(
        f"\nCoverage examples: {len(coverage)}"
    )

    print(
        f"Passage-selection examples: "
        f"{len(passage)}"
    )

    if len(coverage) == 0 or len(passage) == 0:
        print(
            "\nCannot compare: one class "
            "has zero examples."
        )
        return

    rows = []

    for feature in ALL_RUNTIME_FEATURES:

        coverage_values = pd.to_numeric(
            coverage[feature],
            errors="coerce",
        )

        passage_values = pd.to_numeric(
            passage[feature],
            errors="coerce",
        )

        coverage_mean = (
            coverage_values.mean()
        )

        passage_mean = (
            passage_values.mean()
        )

        smd = standardized_mean_difference(
            coverage_values,
            passage_values,
        )

        rows.append(
            {
                "feature": feature,
                "coverage_mean": coverage_mean,
                "passage_mean": passage_mean,
                "abs_standardized_mean_diff": smd,
            }
        )

    result = pd.DataFrame(
        rows
    )

    result = result.sort_values(
        by="abs_standardized_mean_diff",
        ascending=False,
        na_position="last",
    ).reset_index(
        drop=True
    )

    result[
        [
            "coverage_mean",
            "passage_mean",
            "abs_standardized_mean_diff",
        ]
    ] = result[
        [
            "coverage_mean",
            "passage_mean",
            "abs_standardized_mean_diff",
        ]
    ].round(4)

    print(
        "\nFeature separation ranked by "
        "absolute standardized mean difference:"
    )

    print()

    print(
        result.to_string(
            index=False
        )
    )

    print(
        "\nTop 10 descriptive separators:"
    )

    print()

    print(
        result.head(10).to_string(
            index=False
        )
    )


# =========================================================
# Severe retrieval audit
# =========================================================

def audit_rare_classes(
    name: str,
    failures: pd.DataFrame,
) -> None:

    print("\n" + "=" * 100)
    print(
        f"{name}: RARE-CLASS FEASIBILITY"
    )
    print("=" * 100)

    counts = (
        failures[ORACLE_COLUMN]
        .value_counts()
    )

    for failure_type in sorted(
        counts.index
    ):
        count = int(
            counts[failure_type]
        )

        print(
            f"\n{failure_type}: "
            f"{count} examples"
        )

        if count < 10:
            print(
                "  WARNING: too few examples "
                "for a reliable learned class."
            )

        elif count < 30:
            print(
                "  CAUTION: small class; "
                "learned estimates will have "
                "high uncertainty."
            )

        else:
            print(
                "  Sample size is large enough "
                "to justify exploratory modeling."
            )


# =========================================================
# Combined development audit
# =========================================================

def audit_combined_development(
    train_failures: pd.DataFrame,
    validation_failures: pd.DataFrame,
) -> None:

    combined = pd.concat(
        [
            train_failures.assign(
                development_source="train"
            ),
            validation_failures.assign(
                development_source="validation"
            ),
        ],
        ignore_index=True,
    )

    print("\n" + "=" * 100)
    print(
        "COMBINED TRAIN + VALIDATION "
        "FAILURE COUNTS"
    )
    print("=" * 100)

    table = pd.crosstab(
        combined[ORACLE_COLUMN],
        combined["development_source"],
        margins=True,
    )

    print()
    print(
        table.to_string()
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Combined counts are descriptive only."
    )

    print(
        "Validation examples must not be folded "
        "into training for model selection."
    )


# =========================================================
# Main
# =========================================================

def main() -> None:

    print("=" * 100)

    print(
        "PHASE 9.8A — FINE-GRAINED "
        "FAILURE-DIAGNOSIS FEASIBILITY AUDIT"
    )

    print("=" * 100)

    print(
        "\nPurpose:"
    )

    print(
        "Audit class support and runtime-feature "
        "separation before training any "
        "fine-grained failure classifier."
    )

    print(
        "\nNo classifier is trained in this phase."
    )

    print(
        "No threshold is selected."
    )

    print(
        "The held-out test split is not loaded."
    )

    # -----------------------------------------------------
    # Load development splits only
    # -----------------------------------------------------

    print(
        "\nLoading train and validation..."
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

    train_failures = failure_only(
        train_df
    )

    validation_failures = failure_only(
        validation_df
    )

    print(
        f"\nTrain failures: "
        f"{len(train_failures):,}"
    )

    print(
        f"Validation failures: "
        f"{len(validation_failures):,}"
    )

    # -----------------------------------------------------
    # Distribution
    # -----------------------------------------------------

    print_distribution(
        "TRAIN",
        train_failures,
    )

    print_distribution(
        "VALIDATION",
        validation_failures,
    )

    audit_combined_development(
        train_failures,
        validation_failures,
    )

    # -----------------------------------------------------
    # Feature separation
    # -----------------------------------------------------

    audit_coverage_vs_passage(
        "TRAIN",
        train_failures,
    )

    audit_coverage_vs_passage(
        "VALIDATION",
        validation_failures,
    )

    # -----------------------------------------------------
    # Rare classes
    # -----------------------------------------------------

    audit_rare_classes(
        "TRAIN",
        train_failures,
    )

    audit_rare_classes(
        "VALIDATION",
        validation_failures,
    )

    # -----------------------------------------------------
    # Completion
    # -----------------------------------------------------

    print("\n" + "=" * 100)

    print(
        "PHASE 9.8A COMPLETE"
    )

    print("=" * 100)

    print(
        "\nInterpretation rules:"
    )

    print(
        "1. Feature separation is descriptive, "
        "not classifier performance."
    )

    print(
        "2. Large standardized differences may "
        "be unstable for small classes."
    )

    print(
        "3. Severe-retrieval examples must not "
        "be treated as a learnable class if "
        "sample support is inadequate."
    )

    print(
        "4. Validation remains evaluation data, "
        "not extra training data."
    )

    print(
        "5. The held-out test split was not "
        "loaded or inspected."
    )


if __name__ == "__main__":
    main()