
"""Phase 11.4G-12H-10B: HotpotQA question structure discovery.

Read-only diagnostic for understanding real question patterns.
No extraction rules are modified.
"""

from __future__ import annotations

from collections import Counter

from datasets import load_dataset

from src.recovery.title_span_extractor import (
    ConservativeTitleSpanExtractor,
)


DATASET_NAME = "hotpotqa/hotpot_qa"
CONFIG = "distractor"
SPLIT = "validation"

MAX_EXAMPLES = 100
DISPLAY_LIMIT = 25


def main() -> None:
    dataset = load_dataset(
        DATASET_NAME,
        CONFIG,
        split=SPLIT,
    )

    dataset = dataset.select(
        range(min(MAX_EXAMPLES, len(dataset)))
    )

    extractor = ConservativeTitleSpanExtractor()

    question_types = Counter()
    question_starters = Counter()
    matched = 0

    print("=" * 72)
    print("PHASE 11.4G-12H-10B — HOTPOTQA QUESTION STRUCTURE DISCOVERY")
    print("=" * 72)

    for index, record in enumerate(dataset):
        question = record["question"].strip()
        question_type = record["type"]

        question_types[question_type] += 1

        first_words = " ".join(
            question.lower().split()[:3]
        )

        question_starters[first_words] += 1

        candidates = extractor.enumerate_candidates(question)

        if candidates:
            matched += 1

        if index < DISPLAY_LIMIT:
            print(f"\n[{index + 1}]")
            print(f"Type: {question_type}")
            print(f"Question: {question}")
            print(
                "Context titles:",
                record["context"]["title"][:5],
            )
            print(
                "Extractor candidates:",
                [candidate.title for candidate in candidates],
            )

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(f"Questions examined: {len(dataset)}")
    print(f"Questions with extraction candidates: {matched}")

    print("\nQuestion types:")
    for name, count in question_types.most_common():
        print(f"  {name}: {count}")

    print("\nMost common three-word question starters:")
    for starter, count in question_starters.most_common(15):
        print(f"  {starter}: {count}")


if __name__ == "__main__":
    main()
