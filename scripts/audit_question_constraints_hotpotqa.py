
"""Phase 11.4G-12H-11B: HotpotQA constraint signal audit.

Read-only diagnostic on an already inspected development slice.

Counts regex surface signals, not verified semantic constraints.
"""

from __future__ import annotations

from collections import Counter

from datasets import load_dataset

from src.recovery.question_constraint_inventory import (
    QuestionConstraintInventory,
)


START_INDEX = 300
MAX_EXAMPLES = 500
MAX_EXAMPLES_PER_KIND = 5


def main() -> None:
    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
    )

    end_index = min(START_INDEX + MAX_EXAMPLES, len(dataset))

    if START_INDEX >= end_index:
        raise ValueError("Empty evaluation slice.")

    dataset = dataset.select(range(START_INDEX, end_index))

    inventory = QuestionConstraintInventory()

    question_counts = Counter()
    occurrence_counts = Counter()
    examples = {}

    questions_with_signals = 0
    questions_with_multiple_kinds = 0
    questions_with_multiple_occurrences = 0

    for index, record in enumerate(dataset, start=START_INDEX):
        question = record["question"]
        result = inventory.analyze(question)

        if result.has_signals:
            questions_with_signals += 1

        kinds = {signal.kind.value for signal in result.signals}

        if len(kinds) > 1:
            questions_with_multiple_kinds += 1

        if len(result.signals) > 1:
            questions_with_multiple_occurrences += 1

        for kind in kinds:
            question_counts[kind] += 1

        for signal in result.signals:
            kind = signal.kind.value
            occurrence_counts[kind] += 1

            examples.setdefault(kind, [])

            if len(examples[kind]) < MAX_EXAMPLES_PER_KIND:
                examples[kind].append(
                    (
                        index,
                        question,
                        signal.text,
                        signal.start,
                        signal.end,
                    )
                )

    total = len(dataset)

    print("=" * 72)
    print("PHASE 11.4G-12H-11B — HOTPOTQA CONSTRAINT AUDIT")
    print("=" * 72)

    print(f"Questions evaluated: {total}")
    print(f"Questions with signals: {questions_with_signals}")
    print(
        "Questions with multiple kinds: "
        f"{questions_with_multiple_kinds}"
    )
    print(
        "Questions with multiple occurrences: "
        f"{questions_with_multiple_occurrences}"
    )

    print("\nSIGNAL COUNTS")

    for kind in sorted(question_counts):
        print(
            f"{kind:15s} "
            f"questions={question_counts[kind]:4d} "
            f"occurrences={occurrence_counts[kind]:4d}"
        )

    print("\nEXAMPLES")

    for kind in sorted(examples):
        print(f"\n[{kind.upper()}]")

        for index, question, text, start, end in examples[kind]:
            print(f"Index: {index}")
            print(f"Question: {question}")
            print(f"Matched: {text!r} [{start}:{end}]")
            print()

    print("\nNOTE:")
    print(
        "Surface matches are not semantic labels. "
        "No decomposition acceptance is authorized."
    )


if __name__ == "__main__":
    main()
