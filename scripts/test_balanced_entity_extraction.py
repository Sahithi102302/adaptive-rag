from __future__ import annotations

from dataclasses import dataclass

from src.recovery.entity_extraction import (
    ContentEntityExtractor,
    EntityExtractionConfig,
)


@dataclass
class FakeChunk:
    id: str
    text: str


@dataclass
class FakeResult:
    rank: int
    chunk: FakeChunk


class FakeNER:
    """
    Return deterministic entities based on source passage.
    """

    def __call__(
        self,
        text: str,
    ):
        if text == "source-one":
            return [
                {
                    "word": f"Entity A{i}",
                    "entity_group": "ORG",
                    "score": 0.99,
                }
                for i in range(1, 9)
            ]

        if text == "source-two":
            return [
                {
                    "word": f"Entity B{i}",
                    "entity_group": "PER",
                    "score": 0.98,
                }
                for i in range(1, 9)
            ]

        if text == "source-three":
            return [
                {
                    "word": f"Entity C{i}",
                    "entity_group": "LOC",
                    "score": 0.97,
                }
                for i in range(1, 9)
            ]

        return []


def main() -> None:

    print("=" * 80)
    print(
        "PHASE 10.3G-A — PASSAGE-BALANCED "
        "ENTITY EXTRACTION TEST"
    )
    print("=" * 80)

    results = [
        FakeResult(
            rank=1,
            chunk=FakeChunk(
                id="chunk-1",
                text="source-one",
            ),
        ),
        FakeResult(
            rank=2,
            chunk=FakeChunk(
                id="chunk-2",
                text="source-two",
            ),
        ),
        FakeResult(
            rank=3,
            chunk=FakeChunk(
                id="chunk-3",
                text="source-three",
            ),
        ),
    ]

    # --------------------------------------------------------
    # Original behavior
    # --------------------------------------------------------

    original_extractor = (
        ContentEntityExtractor(
            config=EntityExtractionConfig(
                source_result_depth=3,
                max_entities=15,
                max_entities_per_source=None,
            ),
            ner_pipeline=FakeNER(),
        )
    )

    original = (
        original_extractor
        .extract_from_results(
            results
        )
    )

    print(
        "\nOriginal global-budget distribution:"
    )

    for entity in original:
        print(
            f"  rank={entity.source_rank} "
            f"entity={entity.text!r}"
        )

    original_counts = {
        rank: sum(
            entity.source_rank == rank
            for entity in original
        )
        for rank in (1, 2, 3)
    }

    print(
        f"\nOriginal counts: "
        f"{original_counts}"
    )

    assert len(original) == 15

    assert original_counts == {
        1: 8,
        2: 7,
        3: 0,
    }

    print(
        "PASS: original global-budget "
        "behavior is preserved."
    )

    # --------------------------------------------------------
    # Passage-balanced behavior
    # --------------------------------------------------------

    balanced_extractor = (
        ContentEntityExtractor(
            config=EntityExtractionConfig(
                source_result_depth=3,
                max_entities=15,
                max_entities_per_source=5,
            ),
            ner_pipeline=FakeNER(),
        )
    )

    balanced = (
        balanced_extractor
        .extract_from_results(
            results
        )
    )

    print(
        "\nPassage-balanced distribution:"
    )

    for entity in balanced:
        print(
            f"  rank={entity.source_rank} "
            f"entity={entity.text!r}"
        )

    balanced_counts = {
        rank: sum(
            entity.source_rank == rank
            for entity in balanced
        )
        for rank in (1, 2, 3)
    }

    print(
        f"\nBalanced counts: "
        f"{balanced_counts}"
    )

    assert len(balanced) == 15

    assert balanced_counts == {
        1: 5,
        2: 5,
        3: 5,
    }

    print(
        "PASS: each source passage received "
        "its own candidate budget."
    )

    assert any(
        entity.text == "Entity C1"
        for entity in balanced
    )

    print(
        "PASS: rank-3 candidates are no "
        "longer starved by earlier passages."
    )

    print(
        "\nPHASE 10.3G-A UNIT TEST PASSED."
    )


if __name__ == "__main__":
    main()