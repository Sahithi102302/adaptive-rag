
"""Phase 11.4G-12H-10A: HotpotQA title-evidence audit.

Read-only diagnostic.

Measures whether grammar-derived question title candidates match
HotpotQA context-document titles.

Important:
- A context-document title is not necessarily a work/entity title.
- Matching a title does not validate question semantics.
- This audit does not authorize fallback acceptance.
"""

from __future__ import annotations

from collections import Counter

from datasets import load_dataset

from src.ingestion.hotpotqa import build_hotpotqa_corpus
from src.recovery.title_boundary_validator import (
    EvidenceBackedTitleBoundaryValidator,
    BoundaryStatus,
)


MAX_EXAMPLES = 100

DATASET_NAME = "hotpotqa/hotpot_qa"
CONFIG = "distractor"
SPLIT = "validation"


def main() -> None:
    print("=" * 72)
    print("PHASE 11.4G-12H-10A — REAL CORPUS TITLE-EVIDENCE AUDIT")
    print("=" * 72)

    print("\nLoading HotpotQA...")

    dataset = load_dataset(
        DATASET_NAME,
        CONFIG,
        split=SPLIT,
    )

    dataset = dataset.select(
        range(min(MAX_EXAMPLES, len(dataset)))
    )

    corpus = build_hotpotqa_corpus(dataset)

    validator = EvidenceBackedTitleBoundaryValidator()

    documents_by_id = {
        document.id: document
        for document in corpus.documents
    }

    # Only titles from the current question's context are used.
    # This prevents unrelated global corpus titles from creating
    # misleading matches.
    results = Counter()
    examples_with_candidates = 0
    displayed = 0

    for raw_record, example in zip(dataset, corpus.examples):
        context_titles = raw_record["context"]["title"]

        candidates = validator.extractor.enumerate_candidates(
            example.question
        )

        if candidates:
            examples_with_candidates += 1

        result = validator.validate(
            question=example.question,
            known_titles=context_titles,
        )

        results[result.status.value] += 1
        results[f"reason:{result.reason}"] += 1

        if displayed < 12 and candidates:
            displayed += 1

            print(f"\nExample: {example.id}")
            print(f"Question: {example.question}")
            print(
                "Candidates:",
                [candidate.title for candidate in candidates],
            )
            print(
                "Matched title:",
                result.title,
            )
            print(
                "Status:",
                result.status.value,
            )
            print(
                "Reason:",
                result.reason,
            )

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(f"Questions audited: {len(corpus.examples)}")
    print(f"Unique document versions: {len(documents_by_id)}")
    print(f"Questions with candidates: {examples_with_candidates}")

    for key, value in sorted(results.items()):
        print(f"{key}: {value}")

    print("\nInterpretation:")
    print(
        "Title matches establish context-title overlap only; "
        "they do not establish entity identity or semantic correctness."
    )


if __name__ == "__main__":
    main()
