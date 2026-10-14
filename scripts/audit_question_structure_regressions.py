
"""Phase 11.4G-12H-10F: question-structure regression audit.

Compares the current diagnostic against the earlier broad
'who/which ... or' comparison rule on the same HotpotQA sample.

Diagnostic only; no model or acceptance-policy changes.
"""

from __future__ import annotations

import re
from collections import Counter

from datasets import load_dataset

from src.recovery.question_structure_diagnostic import (
    QuestionStructureDiagnostic,
)

START_INDEX = 100
MAX_EXAMPLES = 200

OLD_OR_PATTERN = re.compile(
    r"\b(?:which|who)\b.+\bor\b.+\?",
    re.IGNORECASE,
)


def main() -> None:
    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
    )

    dataset = dataset.select(
        range(
            START_INDEX,
            min(START_INDEX + MAX_EXAMPLES, len(dataset)),
        )
    )

    diagnostic = QuestionStructureDiagnostic()

    counts = Counter()

    print("=" * 72)
    print("PHASE 11.4G-12H-10F — COMPARISON REGRESSION AUDIT")
    print("=" * 72)

    for record in dataset:
        question = record["question"]
        gold = record["type"]

        if not OLD_OR_PATTERN.search(question):
            continue

        result = diagnostic.analyze(question)

        counts[(gold, result.structure.value)] += 1

        print("\nQuestion:", question)
        print("Gold:", gold)
        print("Current prediction:", result.structure.value)
        print("Reason:", result.reason)
        print("Constraints:", result.constraints)

    print("\nSUMMARY")

    for (gold, prediction), count in sorted(counts.items()):
        print(
            f"Gold={gold:12s} "
            f"Current={prediction:22s} "
            f"Count={count}"
        )


if __name__ == "__main__":
    main()
