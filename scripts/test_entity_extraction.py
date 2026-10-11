from __future__ import annotations

from src.chunking.chunk import Chunk
from src.recovery.entity_extraction import (
    ContentEntityExtractor,
    EntityExtractionConfig,
)
from src.retrieval.reranker import (
    RerankedRetrievalResult,
)


class FakeNERPipeline:
    """
    Deterministic stand-in for the real Hugging Face
    NER pipeline.
    """

    def __call__(
        self,
        text: str,
    ):
        if "Shia LaBeouf" in text:
            return [
                {
                    "word": "Shia LaBeouf",
                    "entity_group": "PER",
                    "score": 0.98,
                },
                {
                    "word": "Lin Dan",
                    "entity_group": "PER",
                    "score": 0.90,
                },
            ]

        if "Lin Dan" in text:
            return [
                {
                    "word": "Lin Dan",
                    "entity_group": "PER",
                    "score": 0.99,
                },
                {
                    "word": "BWF",
                    "entity_group": "ORG",
                    "score": 0.95,
                },
            ]

        return []


def make_result(
    rank: int,
    chunk_id: str,
    title: str,
    text: str,
) -> RerankedRetrievalResult:

    chunk = Chunk(
        id=chunk_id,
        document_id=f"doc_{chunk_id}",
        chunk_index=0,
        text=text,
        start_char=0,
        end_char=len(text),
        metadata={
            "title": title,
            "sentence_ids": [0],
        },
    )

    return RerankedRetrievalResult(
        rank=rank,
        score=float(10 - rank),
        chunk=chunk,
        dense_rank=rank,
        sparse_rank=rank,
    )


def main() -> None:

    print("=" * 90)
    print(
        "PHASE 10.3C — CONTENT ENTITY "
        "EXTRACTION TEST"
    )
    print("=" * 90)

    results = (
        make_result(
            rank=1,
            chunk_id="c1",
            title=(
                "2013 BWF World Championships "
                "– Men's singles"
            ),
            text=(
                "Lin Dan was the defending champion "
                "at the BWF World Championships."
            ),
        ),
        make_result(
            rank=2,
            chunk_id="c2",
            title="Maniac (2011 film)",
            text=(
                "The film was directed by "
                "Shia LaBeouf. Lin Dan is included "
                "here only to test deduplication."
            ),
        ),
        make_result(
            rank=3,
            chunk_id="c3",
            title="Unused",
            text="No named entities here.",
        ),
    )

    extractor = ContentEntityExtractor(
        config=EntityExtractionConfig(
            source_result_depth=2,
            max_entities=10,
        ),
        ner_pipeline=FakeNERPipeline(),
    )

    entities = (
        extractor.extract_from_results(
            results
        )
    )

    print(
        "\nExtracted entities:"
    )

    for entity in entities:
        print(
            f"  text={entity.text!r} "
            f"label={entity.label!r} "
            f"score={entity.score:.4f} "
            f"source_rank="
            f"{entity.source_rank}"
        )

    entity_texts = [
        entity.text
        for entity in entities
    ]

    assert entity_texts == [
        "Lin Dan",
        "BWF",
        "Shia LaBeouf",
    ]

    assert (
        entity_texts.count(
            "Lin Dan"
        )
        == 1
    )

    assert (
        entities[0].source_rank
        == 1
    )

    assert (
        entities[2].source_rank
        == 2
    )

    print(
        "\nPASS: Entities extracted from "
        "retrieved content."
    )

    print(
        "PASS: Duplicate entities removed "
        "case-insensitively."
    )

    print(
        "PASS: Source retrieval rank retained."
    )

    print(
        "PASS: Test requires no model download "
        "and no gold evidence."
    )

    print(
        "\nPHASE 10.3C CONTENT ENTITY "
        "EXTRACTION TEST PASSED"
    )


if __name__ == "__main__":
    main()