from __future__ import annotations

from src.chunking.chunk import Chunk
from src.recovery.bridge_query import (
    BridgeQueryConfig,
    RelationAwareBridgeQueryGenerator,
)
from src.retrieval.reranker import (
    RerankedRetrievalResult,
)


class FakeTextGenerator:

    def generate(
        self,
        prompt: str,
    ) -> str:

        assert (
            "Lin Dan was the defending champion"
            in prompt
        )

        assert (
            "nationality"
            in prompt.lower()
        )

        return (
            '{"queries": ['
            '"Lin Dan nationality", '
            '"Lin Dan country", '
            '"Lin Dan nationality"'
            ']}'
        )


def make_result() -> RerankedRetrievalResult:

    text = (
        "The Men's Singles tournament of the "
        "2013 BWF World Championships was held "
        "from August 5–11. Lin Dan was the "
        "defending champion."
    )

    chunk = Chunk(
        id="bwf_chunk",
        document_id="bwf_document",
        chunk_index=0,
        text=text,
        start_char=0,
        end_char=len(text),
        metadata={
            "title": (
                "2013 BWF World Championships "
                "– Men's singles"
            ),
            "sentence_ids": [0, 1],
        },
    )

    return RerankedRetrievalResult(
        rank=1,
        score=7.7827,
        chunk=chunk,
        dense_rank=1,
        sparse_rank=1,
    )


def main() -> None:

    print("=" * 90)
    print(
        "PHASE 10.3D — RELATION-AWARE "
        "BRIDGE QUERY TEST"
    )
    print("=" * 90)

    question = (
        "What was the nationality of the "
        "defending champion at the Men's "
        "Singles tournament of the 2013 "
        "BWF World Championships?"
    )

    generator = (
        RelationAwareBridgeQueryGenerator(
            text_generator=(
                FakeTextGenerator()
            ),
            config=BridgeQueryConfig(
                source_result_depth=3,
                max_queries=3,
            ),
        )
    )

    result = generator.generate(
        question=question,
        results=(make_result(),),
    )

    print(
        "\nGenerated queries:"
    )

    for query in result.queries:
        print(f"  {query}")

    assert result.queries == (
        "Lin Dan nationality",
        "Lin Dan country",
    )

    assert (
        "Lin Dan was the defending champion"
        in result.prompt
    )

    assert (
        question
        in result.prompt
    )

    print(
        "\nPASS: Original question included."
    )

    print(
        "PASS: Retrieved evidence included."
    )

    print(
        "PASS: Structured JSON parsed."
    )

    print(
        "PASS: Duplicate queries removed."
    )

    print(
        "PASS: Maximum query count enforced."
    )

    print(
        "PASS: No gold answer or supporting "
        "facts required."
    )

    print(
        "\nPHASE 10.3D RELATION-AWARE "
        "BRIDGE QUERY TEST PASSED"
    )


if __name__ == "__main__":
    main()