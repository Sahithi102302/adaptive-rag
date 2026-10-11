from __future__ import annotations

from src.chunking.chunk import Chunk
from src.recovery.evidence_selection import (
    EvidencePreservingSelector,
    EvidenceSelectionConfig,
    EvidenceSource,
)
from src.retrieval.reranker import (
    RerankedRetrievalResult,
)


def make_result(
    chunk_id: str,
    rank: int,
) -> RerankedRetrievalResult:

    text = f"Evidence for {chunk_id}."

    chunk = Chunk(
        id=chunk_id,
        document_id=f"doc_{chunk_id}",
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        text=text,
        metadata={
            "title": chunk_id,
        },
    )

    return RerankedRetrievalResult(
        rank=rank,
        score=float(10 - rank),
        chunk=chunk,
    )


def main() -> None:
    print(
        "PHASE 10.4A — EVIDENCE SELECTION TEST"
    )

    first = [
        make_result("A", 1),
        make_result("B", 2),
        make_result("C", 3),
        make_result("D", 4),
        make_result("E", 5),
    ]

    second = [
        make_result("C", 1),
        make_result("F", 2),
        make_result("G", 3),
        make_result("H", 4),
        make_result("I", 5),
    ]

    selector = EvidencePreservingSelector(
        EvidenceSelectionConfig(
            final_k=6,
            reserved_first_hop=3,
            reserved_second_hop=3,
        )
    )

    result = selector.select(
        original_results=first,
        recovered_results=second,
    )

    selected_ids = [
        item.chunk_id
        for item in result.selected_results
    ]

    assert selected_ids == [
        "A", "B", "C", "F", "G", "D"
    ], selected_ids

    print(
        "PASS: deterministic hop-balanced selection."
    )

    assert len(selected_ids) == len(
        set(selected_ids)
    )

    print(
        "PASS: cross-hop deduplication."
    )

    assert (
        result.selected_results[2].source
        == EvidenceSource.BOTH
    )

    assert (
        result.selected_results[2].first_hop_rank
        == 3
    )

    assert (
        result.selected_results[2].second_hop_rank
        == 1
    )

    print(
        "PASS: overlapping chunk provenance preserved."
    )

    assert result.original_results == tuple(
        first
    )

    assert result.recovered_results == tuple(
        second
    )

    print(
        "PASS: both input result sets preserved."
    )

    # Empty second hop should fall back to first-hop evidence.
    fallback = selector.select(
        original_results=first,
        recovered_results=[],
    )

    assert [
        item.chunk_id
        for item in fallback.selected_results
    ] == [
        "A", "B", "C", "D", "E"
    ]

    print(
        "PASS: empty recovery falls back to first hop."
    )

    # Invalid allocation must be rejected.
    try:
        EvidenceSelectionConfig(
            final_k=5,
            reserved_first_hop=3,
            reserved_second_hop=3,
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Expected invalid slot allocation to fail."
        )

    print(
        "PASS: invalid allocation rejected."
    )

    print(
        "\nPHASE 10.4A UNIT TEST PASSED."
    )


if __name__ == "__main__":
    main()