from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.diagnosis.runtime import (
    RUNTIME_FEATURES,
    RuntimeSufficiencyDiagnoser,
)


ARTIFACT_DIR = Path(
    "artifacts/diagnosis/scaled"
)

TRAIN_PATH = (
    ARTIFACT_DIR
    / "diagnostic_train.csv"
)

VALIDATION_PATH = (
    ARTIFACT_DIR
    / "diagnostic_validation.csv"
)

TARGET_COLUMN = (
    "is_evidence_sufficient"
)


def main() -> None:

    print("=" * 100)
    print(
        "PHASE 9.8C — RUNTIME "
        "DIAGNOSER SMOKE TEST"
    )
    print("=" * 100)

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    validation_df = pd.read_csv(
        VALIDATION_PATH
    )

    print(
        f"\nTrain rows: {len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    diagnoser = (
        RuntimeSufficiencyDiagnoser()
    )

    diagnoser.fit(
        feature_frame=train_df[
            RUNTIME_FEATURES
        ],
        labels=train_df[
            TARGET_COLUMN
        ],
    )

    print(
        f"\nFitted: "
        f"{diagnoser.is_fitted}"
    )

    print(
        f"Threshold: "
        f"{diagnoser.threshold:.2f}"
    )

    print(
        f"Runtime features: "
        f"{len(RUNTIME_FEATURES)}"
    )

    # Find one oracle-sufficient and one
    # oracle-insufficient validation example.
    #
    # Gold is used ONLY here to select examples
    # for this offline smoke test. It is not
    # passed into the diagnoser.

    sufficient_row = (
        validation_df[
            validation_df[
                TARGET_COLUMN
            ] == 1
        ]
        .iloc[0]
    )

    insufficient_row = (
        validation_df[
            validation_df[
                TARGET_COLUMN
            ] == 0
        ]
        .iloc[0]
    )

    examples = [
        (
            "oracle-sufficient example",
            sufficient_row,
        ),
        (
            "oracle-insufficient example",
            insufficient_row,
        ),
    ]

    for label, row in examples:

        print(
            "\n" + "-" * 100
        )

        print(label)

        print(
            f"Question index: "
            f"{int(row['question_index'])}"
        )

        print(
            f"Question: "
            f"{row['question']}"
        )

        result = diagnoser.diagnose(
            row[
                RUNTIME_FEATURES
            ]
        )

        print(
            f"P(sufficient): "
            f"{result.sufficient_probability:.4f}"
        )

        print(
            f"Predicted sufficient: "
            f"{result.predicted_sufficient}"
        )

        print(
            f"Failure type: "
            f"{result.diagnosis.failure_type.value}"
        )

        print(
            f"Failure family: "
            f"{result.diagnosis.failure_family.value}"
        )

        print(
            f"Source: "
            f"{result.diagnosis.source.value}"
        )

        print(
            f"Recovery recommended: "
            f"{result.diagnosis.recovery_recommended}"
        )

        print(
            f"Confidence: "
            f"{result.diagnosis.confidence}"
        )

        print(
            "Explanation:"
        )

        print(
            result.diagnosis.explanation
        )

    print(
        "\n" + "=" * 100
    )

    print(
        "PHASE 9.8C SMOKE TEST COMPLETE"
    )

    print("=" * 100)

    print(
        "\nPASS if:"
    )

    print(
        "1. The frozen 29-feature detector fits."
    )

    print(
        "2. Runtime diagnosis executes without "
        "gold-dependent model inputs."
    )

    print(
        "3. Sufficient predictions map to "
        "SUFFICIENT_EVIDENCE."
    )

    print(
        "4. Insufficient predictions map "
        "conservatively to UNKNOWN."
    )

    print(
        "5. Recovery recommendation follows "
        "the binary runtime decision."
    )


if __name__ == "__main__":
    main()