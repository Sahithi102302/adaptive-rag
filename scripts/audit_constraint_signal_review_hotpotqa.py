
"""Phase 11.4G-12H-11D: HotpotQA signal-review audit.

Read-only evaluation on an existing development slice.

Measures diagnostic ambiguity annotations, not semantic accuracy.
Does not authorize decomposition or answer acceptance.
"""

from __future__ import annotations

from collections import Counter

from datasets import load_dataset

from src.recovery.question_constraint_inventory import (
    QuestionConstraintInventory,
)
from src.recovery.constraint_signal_review import (
    ConstraintSignalReviewer,
    SignalReviewStatus,
)


START_INDEX = 300
MAX_EXAMPLES = 500
MAX_EXAMPLES_PER_REASON = 10


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
        raise ValueError("Audit slice is empty.")

    dataset = dataset.select(
        range(START_INDEX, end_index)
    )

    inventory = QuestionConstraintInventory()
    reviewer = ConstraintSignalReviewer()

    status_counts = Counter()
    reason_counts = Counter()
    examples = {}

    total_original_signals = 0
    total_reviewed_signals = 0
    questions_with_ambiguity = 0

    for index, record in enumerate(
        dataset,
        start=START_INDEX,
    ):
        question = record["question"]

        original = inventory.analyze(question)
        reviewed = reviewer.review(original)

        total_original_signals += len(original.signals)
        total_reviewed_signals += len(reviewed.reviewed_signals)

        # Check preservation, including original order and offsets.
        assert reviewed.question == original.question
        assert tuple(
            item.signal for item in reviewed.reviewed_signals
        ) == original.signals, (
            f"Signal preservation failed at index {index}"
        )

        ambiguous_in_question = False

        for item in reviewed.reviewed_signals:
            status_counts[item.status.value] += 1

            if item.status != SignalReviewStatus.POTENTIALLY_AMBIGUOUS:
                continue

            ambiguous_in_question = True
            reason_counts[item.reason] += 1

            examples.setdefault(item.reason, [])

            if len(examples[item.reason]) < MAX_EXAMPLES_PER_REASON:
                examples[item.reason].append(
                    (
                        index,
                        question,
                        item.signal.kind.value,
                        item.signal.text,
                        item.signal.start,
                        item.signal.end,
                    )
                )

        if ambiguous_in_question:
            questions_with_ambiguity += 1

    print("=" * 72)
    print("PHASE 11.4G-12H-11D — HOTPOTQA SIGNAL REVIEW AUDIT")
    print("=" * 72)

    print(f"Questions evaluated: {len(dataset)}")
    print(f"Original signals: {total_original_signals}")
    print(f"Reviewed signals: {total_reviewed_signals}")
    print(f"Questions with ambiguity warnings: {questions_with_ambiguity}")

    print("\nSTATUS COUNTS")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    print("\nAMBIGUITY REASONS")
    if not reason_counts:
        print("  None detected")

    for reason, count in sorted(reason_counts.items()):
        print(f"  {reason}: {count}")

    print("\nAMBIGUITY EXAMPLES")
    if not examples:
        print("  None detected")

    for reason, items in sorted(examples.items()):
        print(f"\n[{reason}]")

        for index, question, kind, text, start, end in items:
            print(f"Index: {index}")
            print(f"Question: {question}")
            print(f"Signal: {kind} / {text!r} [{start}:{end}]")
            print()

    print("\nNOTE:")
    print(
        "Ambiguity warnings are heuristic annotations. "
        "They are not verified semantic interpretations."
    )


if __name__ == "__main__":
    main()
