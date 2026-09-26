from datasets import load_dataset

from src.chunking.sentence_chunker import (
    SentenceChunker,
    SentenceChunkingConfig,
)
from src.ingestion.hotpotqa import (
    build_hotpotqa_corpus,
)
from src.retrieval.dense_retriever import (
    format_chunk_for_retrieval,
)
from src.retrieval.sparse_retriever import (
    BM25Retriever,
)


NUM_EXAMPLES = 100
TOP_K = 10


def get_gold_keys(
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


def get_gold_matches(
    chunk,
    gold_keys: set[tuple[str, int]],
) -> list[tuple[str, int]]:

    sentence_ids = chunk.metadata.get(
        "sentence_ids",
        [],
    )

    return [
        (
            chunk.document_id,
            sentence_id,
        )
        for sentence_id in sentence_ids
        if (
            chunk.document_id,
            sentence_id,
        )
        in gold_keys
    ]


def main() -> None:

    print("=" * 90)
    print(
        "PHASE 7.1 — BM25 SPARSE "
        "RETRIEVAL VALIDATION"
    )
    print("=" * 90)

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
    # Sentence chunking
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

    # --------------------------------------------------------
    # Same retrieval representation as dense retrieval
    # --------------------------------------------------------

    retrieval_texts = [
        format_chunk_for_retrieval(
            chunk
        )
        for chunk in chunks
    ]

    # --------------------------------------------------------
    # Build BM25 index
    # --------------------------------------------------------

    print(
        "\nBuilding BM25 index..."
    )

    retriever = BM25Retriever(
        chunks=chunks,
        retrieval_texts=retrieval_texts,
    )

    print(
        f"Indexed chunks: "
        f"{retriever.size:,}"
    )

    print(
        "Average document length: "
        f"{retriever.average_document_length:.2f} tokens"
    )

    assert retriever.size == len(
        chunks
    )

    # --------------------------------------------------------
    # Use the same first HotpotQA question
    # used for dense validation
    # --------------------------------------------------------

    example = corpus.examples[0]

    gold_keys = get_gold_keys(
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
        "\nGold evidence:"
    )

    for fact in example.supporting_facts:

        print(
            f"  title={fact.title!r}, "
            f"document_id={fact.document_id}, "
            f"sentence_id={fact.sentence_id}"
        )

    # --------------------------------------------------------
    # BM25 retrieval
    # --------------------------------------------------------

    print(
        f"\nRetrieving top-{TOP_K} "
        "BM25 chunks..."
    )

    results = retriever.search(
        query=example.question,
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

    # --------------------------------------------------------
    # Inspect results
    # --------------------------------------------------------

    retrieved_gold: set[
        tuple[str, int]
    ] = set()

    print(
        "\n"
        + "=" * 90
    )

    print(
        "BM25 RETRIEVAL RESULTS"
    )

    print(
        "=" * 90
    )

    for result in results:

        chunk = result.chunk

        matches = get_gold_matches(
            chunk,
            gold_keys,
        )

        retrieved_gold.update(
            matches
        )

        title = chunk.metadata.get(
            "title",
            "<unknown>",
        )

        sentence_ids = chunk.metadata.get(
            "sentence_ids",
            [],
        )

        preview = (
            chunk.text[:250]
            .replace(
                "\n",
                " ",
            )
        )

        print(
            f"\nRank {result.rank}"
        )

        print(
            f"Score: "
            f"{result.score:.6f}"
        )

        print(
            f"Title: "
            f"{title}"
        )

        print(
            f"Chunk ID: "
            f"{chunk.id}"
        )

        print(
            f"Sentence IDs: "
            f"{sentence_ids}"
        )

        print(
            "Gold evidence match: "
            f"{bool(matches)}"
        )

        if matches:
            print(
                f"Matched gold keys: "
                f"{matches}"
            )

        print(
            f"Text: "
            f"{preview}"
        )

    # --------------------------------------------------------
    # Evidence coverage
    # --------------------------------------------------------

    found = (
        gold_keys
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
        f"Gold evidence facts: "
        f"{len(gold_keys)}"
    )

    print(
        f"Gold facts retrieved: "
        f"{len(found)}"
    )

    if gold_keys:

        print(
            "Evidence coverage: "
            f"{len(found) / len(gold_keys):.2%}"
        )

    missing = (
        gold_keys
        - retrieved_gold
    )

    if missing:

        print(
            "\nMissing gold evidence:"
        )

        for key in sorted(
            missing
        ):
            print(
                f"  {key}"
            )

    else:

        print(
            "\nAll gold evidence was "
            f"retrieved in top-{TOP_K}."
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
        "\nPASS: BM25 index built successfully."
    )

    print(
        "PASS: Query -> BM25 scoring -> "
        "Chunk mapping is valid."
    )

    print(
        "PASS: BM25 chunks can be compared "
        "against sentence-level gold evidence."
    )

    print(
        "\nNOTE: Retrieval quality is an "
        "observed result, not a validation condition."
    )


if __name__ == "__main__":
    main()