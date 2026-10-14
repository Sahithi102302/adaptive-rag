
"""Phase 11.4G-12H-10D: HotpotQA question structure evaluation.

Read-only diagnostic evaluation against HotpotQA question-type labels.

The first 100 validation questions were already inspected during
development. This script evaluates the next 200 questions as a
separate diagnostic sample, NOT a final held-out benchmark.

No verifier or fallback acceptance behavior is changed.
"""

from __future__ import annotations

from collections import Counter

from datasets import load_dataset

from src.recovery.question_structure_diagnostic import (
    QuestionStructure,
    QuestionStructureDiagnostic,
)


DATASET_NAME = "hotpotqa/hotpot_qa"
CONFIG = "distractor"
SPLIT = "validation"

START_INDEX = 100
MAX_EXAMPLES = 200
DISPLAY_ERRORS = 15


def main() -> None:
    dataset = load_dataset(
        DATASET_NAME,
        CONFIG,
        split=SPLIT,
    )

    end_index = min(START_INDEX + MAX_EXAMPLES, len(dataset))

    if START_INDEX >= end_index:
        raise ValueError("Requested evaluation range is empty.")

    dataset = dataset.select(range(START_INDEX, end_index))

    diagnostic = QuestionStructureDiagnostic()

    confusion = Counter()
    predicted_counts = Counter()

    covered = 0
    correct_covered = 0
    displayed_errors = 0

    print("=" * 72)
    print("PHASE 11.4G-12H-10D — HOTPOTQA STRUCTURE EVALUATION")
    print("=" * 72)

    for record in dataset:
        question = record["question"]
        gold_type = record["type"]

        result = diagnostic.analyze(question)

        prediction = result.structure.value

        predicted_counts[prediction] += 1
        confusion[(gold_type, prediction)] += 1

        if result.structure == QuestionStructure.UNRESOLVED:
            continue

        covered += 1

        predicted_type = (
            "comparison"
            if result.structure == QuestionStructure.EXPLICIT_COMPARISON
            else "bridge"
        )

        if predicted_type == gold_type:
            correct_covered += 1
        elif displayed_errors < DISPLAY_ERRORS:
            displayed_errors += 1

            print("\nMISCLASSIFICATION")
            print(f"Question: {question}")
            print(f"Gold type: {gold_type}")
            print(f"Predicted: {prediction}")
            print(f"Reason: {result.reason}")
            print(f"Constraints: {result.constraints}")

    total = len(dataset)
    abstained = total - covered

    coverage = covered / total if total else 0.0

    selective_accuracy = (
        correct_covered / covered
        if covered
        else 0.0
    )

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(f"Questions evaluated: {total}")
    print(f"Covered: {covered}")
    print(f"Abstained: {abstained}")
    print(f"Coverage: {coverage:.2%}")
    print(
        "Accuracy on covered questions: "
        f"{selective_accuracy:.2%}"
    )

    print("\nPredicted structures:")
    for name, count in sorted(predicted_counts.items()):
        print(f"  {name}: {count}")

    print("\nGold type × predicted structure:")
    for (gold, prediction), count in sorted(confusion.items()):
        print(f"  {gold:12s} | {prediction:22s} | {count}")

    print("\nNote:")
    print(
        "Agreement with HotpotQA labels measures question-type "
        "classification only, not semantic decomposition validity."
    )


if __name__ == "__main__":
    main()
