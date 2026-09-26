import numpy as np
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


def get_gold_keys(
    example,
) -> set[tuple[str, int]]:
    """
    Convert HotpotQA supporting facts into exact evidence keys:
        (document_id, sentence_id)
    """

    return {
        (
            fact.document_id,
            fact.sentence_id,
        )
        for fact in example.supporting_facts
    }


def get_gold_matches(
    chunk,
    gold_keys: set[tuple[str, int]],
) -> list[tuple[str, int]]:
    """
    Return gold evidence facts contained inside one retrieved chunk.
    """

    sentence_ids = chunk.metadata.get(
        "sentence_ids",
        [],
    )

    matches: list[tuple[str, int]] = []

    for sentence_id in sentence_ids:
        key = (
            chunk.document_id,
            sentence_id,
        )

        if key in gold_keys:
            matches.append(key)

    return matches


def main() -> None:
    print("=" * 78)
    print(
        "AdaptiveRAG - Exact Dense Retrieval Validation"
    )
    print("=" * 78)

    # ---------------------------------------------------------
    # 1. Build development corpus
    # ---------------------------------------------------------

    print("\nLoading HotpotQA...")

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    corpus = build_hotpotqa_corpus(
        dataset["train"],
        max_examples=100,
    )

    print(
        f"QA examples: "
        f"{len(corpus.examples):,}"
    )

    print(
        f"Document versions: "
        f"{len(corpus.documents):,}"
    )

    # ---------------------------------------------------------
    # 2. Sentence-aware chunking
    # ---------------------------------------------------------

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
        f"Sentence-aware chunks: "
        f"{len(chunks):,}"
    )

    # ---------------------------------------------------------
    # 3. Build retrieval representations
    # ---------------------------------------------------------

    retrieval_texts = [
        format_chunk_for_retrieval(chunk)
        for chunk in chunks
    ]

    print("\nSample retrieval representation:")
    print(retrieval_texts[0][:500])

    # ---------------------------------------------------------
    # 4. Embed every chunk in this DEVELOPMENT corpus
    # ---------------------------------------------------------

    print("\nLoading embedding model...")

    embedding_config = DenseEmbeddingConfig(
        model_name=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        batch_size=32,
        normalize_embeddings=True,
    )

    embedder = DenseEmbedder(
        embedding_config
    )

    print(
        f"Embedding dimension: "
        f"{embedder.embedding_dimension}"
    )

    print(
        f"\nEmbedding "
        f"{len(retrieval_texts):,} chunks..."
    )

    chunk_embeddings = (
        embedder.encode_documents(
            retrieval_texts,
            show_progress_bar=True,
        )
    )

    print(
        f"Embedding matrix: "
        f"{chunk_embeddings.shape}"
    )

    # ---------------------------------------------------------
    # 5. Verify normalization
    # ---------------------------------------------------------

    norms = np.linalg.norm(
        chunk_embeddings,
        axis=1,
    )

    print(
        f"Mean vector norm: "
        f"{norms.mean():.6f}"
    )

    assert np.allclose(
        norms,
        1.0,
        atol=1e-5,
    )

    # ---------------------------------------------------------
    # 6. Build exact FAISS index
    # ---------------------------------------------------------

    print("\nBuilding FAISS IndexFlatIP...")

    retriever = DenseRetriever(
        chunks=chunks,
        embeddings=chunk_embeddings,
    )

    print(
        f"FAISS index vectors: "
        f"{retriever.size:,}"
    )

    print(
        f"FAISS dimension: "
        f"{retriever.embedding_dimension}"
    )

    assert retriever.size == len(chunks)

    # ---------------------------------------------------------
    # 7. Select first real HotpotQA question
    # ---------------------------------------------------------

    example = corpus.examples[0]

    print("\n" + "=" * 78)
    print("QUERY")
    print("=" * 78)

    print(f"ID: {example.id}")
    print(f"Question: {example.question}")
    print(f"Gold answer: {example.answer}")

    gold_keys = get_gold_keys(
        example
    )

    print("\nGold evidence:")

    for fact in example.supporting_facts:
        print(
            f"  title={fact.title!r}, "
            f"document_id={fact.document_id}, "
            f"sentence_id={fact.sentence_id}"
        )

    # ---------------------------------------------------------
    # 8. Dense retrieval
    # ---------------------------------------------------------

    top_k = 10

    print(
        f"\nRetrieving top-{top_k} chunks..."
    )

    results = retriever.search(
        query=example.question,
        embedder=embedder,
        top_k=top_k,
    )

    # ---------------------------------------------------------
    # 9. Inspect retrieved chunks
    # ---------------------------------------------------------

    print("\n" + "=" * 78)
    print("RETRIEVAL RESULTS")
    print("=" * 78)

    retrieved_gold_keys: set[
        tuple[str, int]
    ] = set()

    for result in results:
        chunk = result.chunk

        title = chunk.metadata.get(
            "title",
            "<unknown>"
        )

        sentence_ids = chunk.metadata.get(
            "sentence_ids",
            [],
        )

        matches = get_gold_matches(
            chunk,
            gold_keys,
        )

        retrieved_gold_keys.update(
            matches
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
            f"Gold evidence match: "
            f"{bool(matches)}"
        )

        if matches:
            print(
                f"Matched gold keys: "
                f"{matches}"
            )

        preview = (
            chunk.text[:300]
            .replace("\n", " ")
        )

        print(
            f"Text: "
            f"{preview}"
        )

    # ---------------------------------------------------------
    # 10. Evidence coverage for this query
    # ---------------------------------------------------------

    valid_gold_keys = {
        key
        for key in gold_keys
        if key[1] >= 0
    }

    found_count = len(
        retrieved_gold_keys
        & valid_gold_keys
    )

    total_count = len(
        valid_gold_keys
    )

    print("\n" + "=" * 78)
    print("GOLD EVIDENCE COVERAGE")
    print("=" * 78)

    print(
        f"Gold evidence facts: "
        f"{total_count}"
    )

    print(
        f"Gold facts retrieved in top-{top_k}: "
        f"{found_count}"
    )

    if total_count > 0:
        coverage = (
            found_count
            / total_count
        )

        print(
            f"Evidence coverage: "
            f"{coverage:.2%}"
        )

    missing_gold_keys = (
        valid_gold_keys
        - retrieved_gold_keys
    )

    if missing_gold_keys:
        print(
            "\nGold evidence not retrieved:"
        )

        for key in sorted(
            missing_gold_keys
        ):
            print(f"  {key}")

    else:
        print(
            "\nAll gold evidence was represented "
            f"in the top-{top_k} retrieved chunks."
        )

    # ---------------------------------------------------------
    # 11. Structural validation only
    # ---------------------------------------------------------

    print("\n" + "=" * 78)
    print("FINAL VALIDATION")
    print("=" * 78)

    assert len(results) == top_k

    assert all(
        result.rank == expected_rank
        for expected_rank, result
        in enumerate(
            results,
            start=1,
        )
    )

    print(
        "PASS: Exact FAISS dense index built successfully."
    )

    print(
        "PASS: Query embedding -> FAISS search -> "
        "Chunk mapping is valid."
    )

    print(
        "PASS: Retrieved chunks can be compared "
        "against exact sentence-level gold evidence."
    )

    print(
        "\nNOTE: Gold evidence coverage above is an "
        "observed retrieval result, not a validation "
        "condition."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()