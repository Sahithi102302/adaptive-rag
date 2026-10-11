from __future__ import annotations

from dataclasses import dataclass

from src.chunking.chunk import Chunk
from src.recovery.second_hop import (
    SecondHopRetrievalConfig,
    SecondHopRetriever,
)


@dataclass(frozen=True)
class FakeRetrievalResult:
    rank: int
    score: float
    chunk: Chunk


class FakeReranker:
    def __init__(
        self,
        results,
    ) -> None:
        self.results = list(results)
        self.calls: list[dict] = []

    def search(
        self,
        query,
        embedder,
        top_k=10,
    ):
        self.calls.append(
            {
                "query": query,
                "embedder": embedder,
                "top_k": top_k,
            }
        )

        return self.results[:top_k]


class FakeEmbedder:
    pass


def make_chunk(
    chunk_id: str,
    text: str,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=f"doc_{chunk_id}",
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        text=text,
        metadata={
            "title": f"Title {chunk_id}",
        },
    )


def main() -> None:
    print(
        "PHASE 10.3I-A — "
        "SECOND-HOP RETRIEVAL TEST"
    )

    first_chunk = make_chunk(
        "first_1",
        "First-hop evidence.",
    )

    second_chunk_1 = make_chunk(
        "second_1",
        "Recovered evidence one.",
    )

    second_chunk_2 = make_chunk(
        "second_2",
        "Recovered evidence two.",
    )

    first_hop_results = [
        FakeRetrievalResult(
            rank=1,
            score=0.9,
            chunk=first_chunk,
        )
    ]

    fake_second_hop_results = [
        FakeRetrievalResult(
            rank=1,
            score=0.8,
            chunk=second_chunk_1,
        ),
        FakeRetrievalResult(
            rank=2,
            score=0.7,
            chunk=second_chunk_2,
        ),
    ]

    reranker = FakeReranker(
        fake_second_hop_results
    )

    embedder = FakeEmbedder()

    retriever = SecondHopRetriever(
        reranker=reranker,
        embedder=embedder,
        config=SecondHopRetrievalConfig(
            top_k=2
        ),
    )

    # --------------------------------------------------------
    # Valid retrieval
    # --------------------------------------------------------

    result = retriever.retrieve(
        query=(
            "  What nationality is Lin Dan?  "
        ),
        first_hop_results=first_hop_results,
    )

    assert (
        result.query
        == "What nationality is Lin Dan?"
    )

    print(
        "PASS: second-hop query was normalized."
    )

    # --------------------------------------------------------
    # First-hop preservation
    # --------------------------------------------------------

    assert (
        len(result.first_hop_results)
        == 1
    )

    assert (
        result.first_hop_results[0]
        is first_hop_results[0]
    )

    assert (
        result.first_hop_results[0].chunk.id
        == "first_1"
    )

    print(
        "PASS: first-hop results were preserved."
    )

    # --------------------------------------------------------
    # Independent second-hop results
    # --------------------------------------------------------

    assert (
        len(result.second_hop_results)
        == 2
    )

    assert (
        result.second_hop_results[0].chunk.id
        == "second_1"
    )

    assert (
        result.second_hop_results[1].chunk.id
        == "second_2"
    )

    print(
        "PASS: second-hop results were preserved "
        "separately from first-hop results."
    )

    # --------------------------------------------------------
    # Frozen reranker invocation
    # --------------------------------------------------------

    assert len(reranker.calls) == 1

    call = reranker.calls[0]

    assert (
        call["query"]
        == "What nationality is Lin Dan?"
    )

    assert (
        call["embedder"]
        is embedder
    )

    assert (
        call["top_k"]
        == 2
    )

    print(
        "PASS: second-hop retrieval used the "
        "supplied frozen reranker and embedder."
    )

    # --------------------------------------------------------
    # No implicit merging
    # --------------------------------------------------------

    first_ids = {
        item.chunk.id
        for item
        in result.first_hop_results
    }

    second_ids = {
        item.chunk.id
        for item
        in result.second_hop_results
    }

    assert first_ids == {
        "first_1"
    }

    assert second_ids == {
        "second_1",
        "second_2",
    }

    print(
        "PASS: first-hop and second-hop results "
        "were not implicitly merged."
    )

    # --------------------------------------------------------
    # Empty-query guard
    # --------------------------------------------------------

    try:
        retriever.retrieve(
            query="   ",
            first_hop_results=(
                first_hop_results
            ),
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Expected empty second-hop query "
            "to raise ValueError."
        )

    assert len(reranker.calls) == 1

    print(
        "PASS: empty second-hop query was "
        "rejected before retrieval."
    )

    # --------------------------------------------------------
    # Configuration guard
    # --------------------------------------------------------

    try:
        SecondHopRetrievalConfig(
            top_k=0
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Expected top_k=0 to raise "
            "ValueError."
        )

    print(
        "PASS: invalid top_k was rejected."
    )

    print(
        "\nPHASE 10.3I-A UNIT TEST PASSED."
    )


if __name__ == "__main__":
    main()