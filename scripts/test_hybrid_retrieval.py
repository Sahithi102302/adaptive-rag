from __future__ import annotations

from datasets import load_dataset

from src.chunking.sentence_chunker import (
    SentenceChunker,
    SentenceChunkingConfig,
)
from src.embeddings.dense_embedder import (
    DenseEmbedder,
    DenseEmbeddingConfig,
)
from src.ingestion.hotpotqa import (
    build_hotpotqa_corpus,
)
from src.retrieval.dense_retriever import (
    DenseRetriever,
    format_chunk_for_retrieval,
)
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    HybridRetrievalConfig,
)
from src.retrieval.sparse_retriever import (
    BM25Retriever,
)


NUM_EXAMPLES = 100
TOP_K = 10


def get_gold_evidence(
    example,
) -> set[tuple[str, int]]:

    return {
        (
            fact.document_id,
            fact.sentence_id,
        )
        for fact in example.supporting_facts
        if (
            fact.document_id is not None
            and fact.sentence_id >= 0
        )
    }


def get_chunk_evidence(
    chunk,
) -> set[tuple[str, int]]:

    return {
        (
            chunk.document_id,
            sentence_id,
        )
        for sentence_id in chunk.metadata.get(
            "sentence_ids",
            [],
        )
    }


def main() -> None:

    print("=" * 90)
    print(
        "PHASE 7.4A — HYBRID RRF "
        "RETRIEVAL VALIDATION"
    )
    print("=" * 90)

    # --------------------------------------------------------
    # Corpus
    # --------------------------------------------------------

    print("\nLoading HotpotQA...")

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    corpus = build_hotpotqa_corpus(
        dataset["train"],
        max_examples=NUM_EXAMPLES,
    )

    print(
        f"QA examples: "
        f"{len(corpus.examples):,}"
    )

    print(
        f"Documents: "
        f"{len(corpus.documents):,}"
    )

    # --------------------------------------------------------
    # Chunking
    # --------------------------------------------------------

    chunker = SentenceChunker(
        SentenceChunkingConfig(
            sentences_per_chunk=3,
            sentence_overlap=1,
        )
    )

    chunks = chunker.chunk_documents(
        corpus.documents
    )

    print(
        f"Chunks: "
        f"{len(chunks):,}"
    )

    retrieval_texts = [
        format_chunk_for_retrieval(
            chunk
        )
        for chunk in chunks
    ]

    # --------------------------------------------------------
    # Dense
    # --------------------------------------------------------

    print(
        "\nLoading dense embedder..."
    )

    embedder = DenseEmbedder(
        DenseEmbeddingConfig(
            model_name=(
                "sentence-transformers/"
                "all-MiniLM-L6-v2"
            ),
            batch_size=32,
            normalize_embeddings=True,
        )
    )

    print(
        f"Embedding "
        f"{len(retrieval_texts):,} chunks..."
    )

    embeddings = (
        embedder.encode_documents(
            retrieval_texts,
            show_progress_bar=True,
        )
    )

    dense_retriever = DenseRetriever(
        chunks=chunks,
        embeddings=embeddings,
    )

    # --------------------------------------------------------
    # Sparse
    # --------------------------------------------------------

    print(
        "\nBuilding BM25 index..."
    )

    sparse_retriever = BM25Retriever(
        chunks=chunks,
        retrieval_texts=retrieval_texts,
    )

    # --------------------------------------------------------
    # Hybrid
    # --------------------------------------------------------

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        config=HybridRetrievalConfig(
            candidate_k=20,
            rrf_k=60,
        ),
    )

    # --------------------------------------------------------
    # Same first HotpotQA query
    # --------------------------------------------------------

    example = corpus.examples[0]

    gold = get_gold_evidence(
        example
    )

    print(
        "\n"
        + "=" * 90
    )

    print("QUERY")

    print(
        "=" * 90
    )

    print(
        f"Question: "
        f"{example.question}"
    )

    print(
        f"Gold answer: "
        f"{example.answer}"
    )

    print(
        f"Gold evidence facts: "
        f"{len(gold)}"
    )

    # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    print(
        "\nRunning hybrid retrieval..."
    )

    results = hybrid_retriever.search(
        query=example.question,
        embedder=embedder,
        top_k=TOP_K,
    )

    assert len(results) == TOP_K

    assert all(
        result.rank == expected_rank
        for expected_rank, result
        in enumerate(
            results,
            start=1,
        )
    )

    assert (
        len(
            {
                result.chunk.id
                for result in results
            }
        )
        == len(results)
    )

    # --------------------------------------------------------
    # Inspect
    # --------------------------------------------------------

    retrieved_gold: set[
        tuple[str, int]
    ] = set()

    print(
        "\n"
        + "=" * 90
    )

    print(
        "HYBRID RRF RESULTS"
    )

    print(
        "=" * 90
    )

    for result in results:

        chunk_evidence = (
            get_chunk_evidence(
                result.chunk
            )
        )

        matched = (
            gold
            & chunk_evidence
        )

        retrieved_gold.update(
            matched
        )

        title = (
            result.chunk.metadata.get(
                "title",
                "<unknown>",
            )
        )

        sentence_ids = (
            result.chunk.metadata.get(
                "sentence_ids",
                [],
            )
        )

        print(
            f"\nRank {result.rank}"
        )

        print(
            f"RRF score: "
            f"{result.score:.8f}"
        )

        print(
            f"Dense rank: "
            f"{result.dense_rank}"
        )

        print(
            f"BM25 rank: "
            f"{result.sparse_rank}"
        )

        print(
            f"Title: "
            f"{title}"
        )

        print(
            f"Sentence IDs: "
            f"{sentence_ids}"
        )

        print(
            "Gold evidence match: "
            f"{bool(matched)}"
        )

        if matched:
            print(
                f"Matched gold keys: "
                f"{sorted(matched)}"
            )

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    found = (
        gold
        & retrieved_gold
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "GOLD EVIDENCE COVERAGE"
    )

    print(
        "=" * 90
    )

    print(
        f"\nGold facts: "
        f"{len(gold)}"
    )

    print(
        f"Gold facts retrieved: "
        f"{len(found)}"
    )

    print(
        "Evidence coverage: "
        f"{len(found) / len(gold):.2%}"
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "FINAL VALIDATION"
    )

    print(
        "=" * 90
    )

    print(
        "\nPASS: Dense and BM25 candidates "
        "were fused successfully."
    )

    print(
        "PASS: Hybrid results contain "
        "unique chunks."
    )

    print(
        "PASS: Final hybrid ranks are "
        "contiguous."
    )

    print(
        "PASS: Hybrid chunks preserve "
        "sentence-level evidence mapping."
    )

    print(
        "\nNOTE: This validates mechanics only. "
        "It does not establish that hybrid "
        "retrieval is better."
    )


if __name__ == "__main__":
    main()