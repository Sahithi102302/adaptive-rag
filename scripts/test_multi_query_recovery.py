from __future__ import annotations

from src.chunking.chunk import Chunk
from src.recovery.multi_query import (
    MultiQueryRecovery,
    MultiQueryRecoveryConfig,
)
from src.recovery.states import (
    RecoveryAction,
    RecoveryRequest,
    RecoveryStatus,
)
from src.retrieval.reranker import (
    RerankedRetrievalResult,
)


def make_result(
    rank: int,
    chunk_id: str,
    title: str,
) -> RerankedRetrievalResult:

    chunk = Chunk(
        id=chunk_id,
        document_id=f"doc_{chunk_id}",
        chunk_index=0,
        text=f"Example text for {title}.",
        start_char=0,
        end_char=20,
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
        "PHASE 10.3A — MULTI-QUERY "
        "RECOVERY TEST"
    )
    print("=" * 90)

    original_results = (
        make_result(
            1,
            "c1",
            "High Bridge (Latah Creek)",
        ),
        make_result(
            2,
            "c2",
            "Spokane, Portland and "
            "Seattle Railway",
        ),
        make_result(
            3,
            "c3",
            "High Bridge (Latah Creek)",
        ),
        make_result(
            4,
            "c4",
            "Great Northern Railway",
        ),
    )

    request = RecoveryRequest(
        original_query=(
            "In what year did the railroad "
            "that built the High Bridge "
            "cease to operate?"
        ),
        action=(
            RecoveryAction
            .MULTI_QUERY_RETRIEVAL
        ),
        original_results=(
            original_results
        ),
        metadata={},
    )

    recovery = MultiQueryRecovery(
        MultiQueryRecoveryConfig(
            max_bridge_titles=3,
            source_result_depth=4,
        )
    )

    result = recovery.recover(
        request
    )

    print(
        "\nStatus:",
        result.status.value,
    )

    print(
        "Succeeded:",
        result.succeeded,
    )

    print(
        "\nRecovery queries:"
    )

    for index, query in enumerate(
        result.recovery_queries,
        start=1,
    ):
        print(
            f"  {index}. {query}"
        )

    print(
        "\nMetadata:",
        result.metadata,
    )

    assert (
        result.status
        == RecoveryStatus.COMPLETED
    )

    assert result.succeeded

    assert (
        result.original_results
        == original_results
    )

    assert (
        result.recovered_results
        == tuple()
    )

    assert (
        len(result.recovery_queries)
        == 4
    )

    assert (
        result.recovery_queries[0]
        == request.original_query
    )

    assert (
        result.recovery_queries[1]
        == (
            request.original_query
            + " High Bridge (Latah Creek)"
        )
    )

    assert (
        result.recovery_queries[2]
        == (
            request.original_query
            + " Spokane, Portland and "
            "Seattle Railway"
        )
    )

    assert (
        result.recovery_queries[3]
        == (
            request.original_query
            + " Great Northern Railway"
        )
    )

    assert (
        result.metadata[
            "retrieval_executed"
        ]
        is False
    )

    print(
        "\nPASS: Original query preserved."
    )

    print(
        "PASS: Duplicate titles removed."
    )

    print(
        "PASS: Bridge-title query expansion "
        "is deterministic."
    )

    print(
        "PASS: No retrieval or gold evidence "
        "was used during query construction."
    )

    print(
        "\nPHASE 10.3A MULTI-QUERY "
        "RECOVERY TEST PASSED"
    )


if __name__ == "__main__":
    main()