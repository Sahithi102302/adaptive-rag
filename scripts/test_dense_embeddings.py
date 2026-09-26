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


def cosine_from_normalized_vectors(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    """
    For L2-normalized vectors, dot product equals cosine similarity.
    """

    return float(
        np.dot(vector_a, vector_b)
    )


def main() -> None:
    print("=" * 70)
    print("AdaptiveRAG - Dense Embedding Validation")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load a small HotpotQA development corpus
    # ---------------------------------------------------------

    print("\nLoading HotpotQA...")

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    train_dataset = dataset["train"]

    corpus = build_hotpotqa_corpus(
        train_dataset,
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
    # 2. Build sentence-aware chunks
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
    # 3. Load embedding model
    # ---------------------------------------------------------

    print("\nLoading dense embedding model...")

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
        f"Model: "
        f"{embedding_config.model_name}"
    )

    print(
        f"Embedding dimension: "
        f"{embedder.embedding_dimension}"
    )

    print(
        f"Normalize embeddings: "
        f"{embedding_config.normalize_embeddings}"
    )

    # ---------------------------------------------------------
    # 4. Embed a manageable chunk sample
    # ---------------------------------------------------------

    sample_chunks = chunks[:100]

    chunk_texts = [
        chunk.text
        for chunk in sample_chunks
    ]

    print(
        f"\nEmbedding "
        f"{len(sample_chunks)} chunks..."
    )

    chunk_embeddings = (
        embedder.encode_documents(
            chunk_texts
        )
    )

    print(
        f"Embedding matrix shape: "
        f"{chunk_embeddings.shape}"
    )

    print(
        f"Embedding dtype: "
        f"{chunk_embeddings.dtype}"
    )

    # ---------------------------------------------------------
    # 5. Validate vector norms
    # ---------------------------------------------------------

    norms = np.linalg.norm(
        chunk_embeddings,
        axis=1,
    )

    print("\nVector norm statistics:")

    print(
        f"  Minimum: "
        f"{norms.min():.6f}"
    )

    print(
        f"  Maximum: "
        f"{norms.max():.6f}"
    )

    print(
        f"  Mean:    "
        f"{norms.mean():.6f}"
    )

    assert np.allclose(
        norms,
        1.0,
        atol=1e-5,
    ), (
        "Embeddings are expected to be "
        "L2-normalized."
    )

    # ---------------------------------------------------------
    # 6. Validate deterministic chunk-vector mapping
    # ---------------------------------------------------------

    print(
        "\nChecking chunk -> vector alignment..."
    )

    assert (
        len(sample_chunks)
        == chunk_embeddings.shape[0]
    )

    for index, chunk in enumerate(
        sample_chunks[:5]
    ):
        print(
            f"  Vector {index:02d} -> "
            f"{chunk.id}"
        )

    # ---------------------------------------------------------
    # 7. Semantic sanity check
    # ---------------------------------------------------------

    print("\nSemantic similarity sanity check...")

    query = (
        "Who is the chief executive officer "
        "of Radio City?"
    )

    relevant_text = (
        "Abraham Thomas is the CEO "
        "of the company."
    )

    unrelated_text = (
        "The magazine was published "
        "in Philadelphia."
    )

    test_texts = [
        query,
        relevant_text,
        unrelated_text,
    ]

    test_embeddings = embedder.encode(
        test_texts
    )

    query_vector = test_embeddings[0]
    relevant_vector = test_embeddings[1]
    unrelated_vector = test_embeddings[2]

    relevant_similarity = (
        cosine_from_normalized_vectors(
            query_vector,
            relevant_vector,
        )
    )

    unrelated_similarity = (
        cosine_from_normalized_vectors(
            query_vector,
            unrelated_vector,
        )
    )

    print(f"\nQuery:")
    print(f"  {query}")

    print("\nRelevant text:")
    print(f"  {relevant_text}")

    print(
        f"  Similarity: "
        f"{relevant_similarity:.6f}"
    )

    print("\nUnrelated text:")
    print(f"  {unrelated_text}")

    print(
        f"  Similarity: "
        f"{unrelated_similarity:.6f}"
    )

    assert (
        relevant_similarity
        > unrelated_similarity
    ), (
        "Semantic sanity check failed: "
        "relevant evidence should score above "
        "the unrelated text."
    )

    # ---------------------------------------------------------
    # 8. Query embedding interface
    # ---------------------------------------------------------

    print(
        "\nTesting single-query interface..."
    )

    single_query_embedding = (
        embedder.encode_query(query)
    )

    print(
        f"Query embedding shape: "
        f"{single_query_embedding.shape}"
    )

    assert (
        single_query_embedding.shape
        == (embedder.embedding_dimension,)
    )

    assert np.isclose(
        np.linalg.norm(
            single_query_embedding
        ),
        1.0,
        atol=1e-5,
    )

    # ---------------------------------------------------------
    # 9. Final result
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    print(
        "PASS: Dense embeddings have the "
        "expected dimensionality."
    )

    print(
        "PASS: Embeddings are L2-normalized."
    )

    print(
        "PASS: Chunk order remains aligned "
        "with vector row order."
    )

    print(
        "PASS: Relevant evidence scored above "
        "the unrelated sanity-check text."
    )

    print(
        "PASS: Single-query embedding interface "
        "is valid."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()