from __future__ import annotations

from pathlib import Path

import pandas as pd


DATASET_PATH = Path(
    "artifacts/diagnosis/diagnostic_dataset_100.csv"
)

TARGET_COLUMN = "is_evidence_sufficient"

NON_FEATURE_COLUMNS = {
    "question_index",
    "question",
    "oracle_failure_type",
    TARGET_COLUMN,
}


def main() -> None:

    print("=" * 90)
    print("PHASE 9.3 — RUNTIME SIGNAL ANALYSIS")
    print("=" * 90)

    # ---------------------------------------------------------
    # Load diagnostic dataset
    # ---------------------------------------------------------

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Diagnostic dataset not found: "
            f"{DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    print(
        f"\nRows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    # ---------------------------------------------------------
    # Validate target
    # ---------------------------------------------------------

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Missing target column: "
            f"{TARGET_COLUMN}"
        )

    target_counts = (
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    print("\nTarget distribution:")

    for value, count in target_counts.items():

        label = (
            "sufficient"
            if value == 1
            else "insufficient"
        )

        print(
            f"  {label}: {count}"
        )

    # ---------------------------------------------------------
    # Select numeric runtime features only
    # ---------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    numeric_features = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    print(
        f"\nNumeric runtime features: "
        f"{len(numeric_features)}"
    )

    # ---------------------------------------------------------
    # Group means
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print("GROUP MEANS")
    print("=" * 90)

    grouped_means = (
        df.groupby(TARGET_COLUMN)[
            numeric_features
        ]
        .mean()
        .T
    )

    grouped_means.columns = [
        "insufficient_mean",
        "sufficient_mean",
    ]

    grouped_means[
        "absolute_mean_difference"
    ] = (
        grouped_means[
            "sufficient_mean"
        ]
        - grouped_means[
            "insufficient_mean"
        ]
    ).abs()

    grouped_means = (
        grouped_means.sort_values(
            "absolute_mean_difference",
            ascending=False,
        )
    )

    print(
        grouped_means.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # ---------------------------------------------------------
    # Point-biserial equivalent:
    # Pearson correlation with binary target
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print("FEATURE / SUFFICIENCY CORRELATIONS")
    print("=" * 90)

    correlations: list[
        tuple[str, float]
    ] = []

    for feature in numeric_features:

        valid = df[
            [feature, TARGET_COLUMN]
        ].dropna()

        if valid[feature].nunique() <= 1:
            correlation = float("nan")

        else:
            correlation = valid[
                feature
            ].corr(
                valid[TARGET_COLUMN]
            )

        correlations.append(
            (
                feature,
                correlation,
            )
        )

    correlation_df = pd.DataFrame(
        correlations,
        columns=[
            "feature",
            "correlation_with_sufficiency",
        ],
    )

    correlation_df[
        "absolute_correlation"
    ] = (
        correlation_df[
            "correlation_with_sufficiency"
        ].abs()
    )

    correlation_df = (
        correlation_df.sort_values(
            "absolute_correlation",
            ascending=False,
            na_position="last",
        )
    )

    print(
        correlation_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ---------------------------------------------------------
    # Missing-value analysis
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print("MISSING VALUES")
    print("=" * 90)

    missing_counts = (
        df[numeric_features]
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(
        missing_counts.to_string()
    )

    # ---------------------------------------------------------
    # Failure-type means
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print("MEANS BY ORACLE FAILURE TYPE")
    print("=" * 90)

    failure_means = (
        df.groupby(
            "oracle_failure_type"
        )[numeric_features]
        .mean()
    )

    print(
        failure_means.T.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # ---------------------------------------------------------
    # Sanity checks
    # ---------------------------------------------------------

    assert len(df) == 100

    assert int(
        df[TARGET_COLUMN].sum()
    ) == 75

    assert (
        len(df)
        - int(df[TARGET_COLUMN].sum())
        == 25
    )

    print("\n" + "=" * 90)

    print(
        "PASS: Runtime signal analysis "
        "completed successfully."
    )

    print(
        "NOTE: These are exploratory "
        "development-slice statistics, "
        "not held-out performance results."
    )


if __name__ == "__main__":
    main()