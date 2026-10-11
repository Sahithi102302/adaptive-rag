from __future__ import annotations

from src.chunking.chunk import Chunk
from src.recovery.bridge_query import (
    BridgeQueryConfig,
    RelationAwareBridgeQueryGenerator,
)
from src.recovery.qwen_generator import (
    QwenGeneratorConfig,
    QwenTextGenerator,
)
from src.retrieval.reranker import (
    RerankedRetrievalResult,
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
        "PHASE 10.3D-REAL — QWEN BRIDGE "
        "QUERY SMOKE TEST"
    )
    print("=" * 90)

    question = (
        "What was the nationality of the "
        "defending champion at the Men's "
        "Singles tournament of the 2013 "
        "BWF World Championships?"
    )

    text_generator = QwenTextGenerator(
        QwenGeneratorConfig(
            max_new_tokens=128,
        )
    )

    bridge_generator = (
        RelationAwareBridgeQueryGenerator(
            text_generator=text_generator,
            config=BridgeQueryConfig(
                source_result_depth=1,
                max_queries=3,
            ),
        )
    )

    result = bridge_generator.generate(
        question=question,
        results=(make_result(),),
    )

    print(
        "\nRaw model output:"
    )
    print(result.raw_output)

    print(
        "\nParsed bridge queries:"
    )

    for query in result.queries:
        print(f"  {query}")

    assert result.queries

    print(
        "\nPASS: Qwen generated valid "
        "structured bridge queries."
    )

    print(
        "PASS: Output parsed through the "
        "model-independent bridge interface."
    )

    print(
        "\nPHASE 10.3D-REAL QWEN "
        "SMOKE TEST PASSED"
    )


if __name__ == "__main__":
    main()