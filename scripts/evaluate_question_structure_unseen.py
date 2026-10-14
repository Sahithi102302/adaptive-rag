
"""Phase 11.4G-12H-10H: unseen-slice question structure evaluation.

Evaluates the frozen question-structure diagnostic on a separate
HotpotQA validation slice.

Do not tune the diagnostic using this slice if it is intended
to remain a held-out evaluation set.

This evaluates question-type agreement, not semantic equivalence
or decomposition correctness.
"""

from __future__ import annotations

from collections import Counter

from datasets import load_dataset

from src.recovery.question_structure_diagnostic import (
    QuestionStructure,
    QuestionStructureDiagnostic,
)


START_INDEX = 300
MAX_EXAMPLES = 500
MAX_DISPLAY_ERRORS = 15


def main() -> None:
    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
    )

    end_index = min(
        START_INDEX + MAX_EXAMPLES,
        len(dataset),
    )

    if START_INDEX >= end_index:
        raise ValueError("Evaluation slice is empty.")

    dataset = dataset.select(
        range(START_INDEX, end_index)
    )

    diagnostic = QuestionStructureDiagnostic()

    confusion = Counter()
    predicted_counts = Counter()

    covered = 0
    correct = 0
    errors_displayed = 0

    print("=" * 72)
    print("PHASE 11.4G-12H-10H — UNSEEN HOTPOTQA EVALUATION")
    print("=" * 72)

    for record in dataset:
        question = record["question"]
        gold = record["type"]

        result = diagnostic.analyze(question)

        predicted_counts[result.structure.value] += 1
        confusion[(gold, result.structure.value)] += 1

        if result.structure == QuestionStructure.UNRESOLVED:
            continue

        covered += 1

        predicted_type = (
            "comparison"
            if result.structure == QuestionStructure.EXPLICIT_COMPARISON
            else "bridge"
        )

        if predicted_type == gold:
            correct += 1
        elif errors_displayed < MAX_DISPLAY_ERRORS:
            errors_displayed += 1

            print("\nMISCLASSIFICATION")
            print(f"Question: {question}")
            print(f"Gold: {gold}")
            print(f"Predicted: {result.structure.value}")
            print(f"Reason: {result.reason}")
            print(f"Constraints: {result.constraints}")

    total = len(dataset)
    abstained = total - covered

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(f"Questions evaluated: {total}")
    print(f"Covered: {covered}")
    print(f"Abstained: {abstained}")
    print(f"Coverage: {covered / total:.2%}")

    if covered:
        print(
            "Selective accuracy: "
            f"{correct / covered:.2%}"
        )
    else:
        print("Selective accuracy: N/A (no predictions)")

    print("\nPredicted structures:")
    for name, count in sorted(predicted_counts.items()):
        print(f"  {name}: {count}")

    print("\nGold type × predicted structure:")
    for (gold, prediction), count in sorted(confusion.items()):
        print(
            f"  {gold:12s} | "
            f"{prediction:22s} | {count}"
        )

    print("\nInterpretation:")
    print(
        "Results measure agreement with HotpotQA question-type "
        "labels, not semantic decomposition correctness."
    )


if __name__ == "__main__":
    main()
