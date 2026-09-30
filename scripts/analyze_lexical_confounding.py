from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATASET_PATH = Path(
    "artifacts/diagnosis/diagnostic_dataset_100.csv"
)

TARGET = "is_evidence_sufficient"
LENGTH = "informative_query_token_count"

LEXICAL_FEATURES = [
    "query_token_coverage",
    "uncovered_query_token_count",
]


def partial_correlation(
    df: pd.DataFrame,
    feature: str,
    target: str,
    control: str,
) -> float:
    """
    Pearson partial correlation between feature and target
    after linearly removing the control variable from both.

    Used here only as exploratory development-slice analysis.
    """

    data = df[
        [feature, target, control]
    ].dropna()

    x = data[feature].to_numpy(dtype=float)
    y = data[target].to_numpy(dtype=float)
    z = data[control].to_numpy(dtype=float)

    design = np.column_stack(
        [
            np.ones(len(z)),
            z,
        ]
    )

    beta_x, *_ = np.linalg.lstsq(
        design,
        x,
        rcond=None,
    )

    beta_y, *_ = np.linalg.lstsq(
        design,
        y,
        rcond=None,
    )

    residual_x = x - design @ beta_x
    residual_y = y - design @ beta_y

    if (
        np.std(residual_x) == 0
        or np.std(residual_y) == 0
    ):
        return float("nan")

    return float(
        np.corrcoef(
            residual_x,
            residual_y,
        )[0, 1]
    )


def main() -> None:

    print("=" * 90)
    print(
        "PHASE 9.4D — QUERY-LENGTH "
        "CONFOUNDING ANALYSIS"
    )
    print("=" * 90)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required = {
        TARGET,
        LENGTH,
        *LEXICAL_FEATURES,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    # ---------------------------------------------------------
    # Basic relationships
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print("RAW CORRELATIONS")
    print("=" * 90)

    columns = [
        LENGTH,
        *LEXICAL_FEATURES,
        TARGET,
    ]

    correlation_matrix = (
        df[columns].corr()
    )

    print(
        correlation_matrix.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # ---------------------------------------------------------
    # Partial correlations controlling for query length
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print(
        "PARTIAL CORRELATIONS WITH SUFFICIENCY"
    )
    print(
        f"Controlling for: {LENGTH}"
    )
    print("=" * 90)

    rows = []

    for feature in LEXICAL_FEATURES:

        raw = df[
            feature
        ].corr(
            df[TARGET]
        )

        partial = partial_correlation(
            df=df,
            feature=feature,
            target=TARGET,
            control=LENGTH,
        )

        rows.append(
            {
                "feature": feature,
                "raw_correlation": raw,
                "partial_correlation": partial,
                "absolute_raw": abs(raw),
                "absolute_partial": abs(partial),
            }
        )

    result_df = pd.DataFrame(rows)

    print(
        result_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ---------------------------------------------------------
    # Normalized uncovered-token rate
    #
    # This should be mathematically related to
    # 1 - query_token_coverage.
    # We calculate it explicitly as a consistency check.
    # ---------------------------------------------------------

    df[
        "uncovered_query_token_fraction"
    ] = (
        df["uncovered_query_token_count"]
        / df[LENGTH]
    )

    expected = (
        1.0
        - df["query_token_coverage"]
    )

    max_difference = (
        df[
            "uncovered_query_token_fraction"
        ]
        - expected
    ).abs().max()

    print("\n" + "=" * 90)
    print("NORMALIZATION CHECK")
    print("=" * 90)

    print(
        "\nMaximum difference between:"
    )

    print(
        "  uncovered_query_token_fraction"
    )

    print(
        "and:"
    )

    print(
        "  1 - query_token_coverage"
    )

    print(
        f"\nDifference: {max_difference:.12f}"
    )

    assert max_difference < 1e-12

    normalized_correlation = (
        df[
            "uncovered_query_token_fraction"
        ].corr(
            df[TARGET]
        )
    )

    print(
        "\nCorrelation of normalized "
        "uncovered-token fraction "
        "with sufficiency:"
    )

    print(
        f"  {normalized_correlation:.4f}"
    )

    # ---------------------------------------------------------
    # Length buckets
    #
    # Descriptive only. With 100 examples these groups may be
    # small and must not be treated as validation results.
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print("QUERY-LENGTH QUARTILES")
    print("=" * 90)

    df["length_bucket"] = pd.qcut(
        df[LENGTH],
        q=4,
        duplicates="drop",
    )

    bucket_summary = (
        df.groupby(
            "length_bucket",
            observed=True,
        )
        .agg(
            questions=(TARGET, "size"),
            mean_query_length=(LENGTH, "mean"),
            sufficiency_rate=(TARGET, "mean"),
            mean_coverage=(
                "query_token_coverage",
                "mean",
            ),
            mean_uncovered=(
                "uncovered_query_token_count",
                "mean",
            ),
        )
    )

    print(
        bucket_summary.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # ---------------------------------------------------------
    # Sanity checks
    # ---------------------------------------------------------

    assert len(df) == 100
    assert int(df[TARGET].sum()) == 75

    print("\n" + "=" * 90)

    print(
        "PASS: Query-length confounding "
        "analysis completed successfully."
    )

    print(
        "NOTE: Partial correlations and "
        "quartiles are exploratory statistics "
        "on the 100-question development slice."
    )


if __name__ == "__main__":
    main()