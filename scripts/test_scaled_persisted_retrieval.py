from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from datasets import load_dataset

from src.chunking.chunk import Chunk
from src.embeddings.dense_embedder import (
    DenseEmbedder,
    DenseEmbeddingConfig,
)
from src.retrieval.dense_retriever import (
    DenseRetriever,
)


ARTIFACT_DIR = Path(
    "artifacts/diagnosis/scaled"
)

CHUNKS_PATH = (
    ARTIFACT_DIR
    / "pooled_chunks.jsonl"
)

EMBEDDINGS_PATH = (
    ARTIFACT_DIR
    / "pooled_embeddings.npy"
)

FAISS_INDEX_PATH = (
    ARTIFACT_DIR
    / "pooled_faiss.index"
)


MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

EXPECTED_CHUNKS = 32368
EXPECTED_DIMENSION = 384

TOP_K = 10


def load_chunks() -> list[Chunk]:
    """
    Load frozen chunks while preserving JSONL row order.

    Row i must remain aligned with embedding row i and
    FAISS vector i.
    """

    chunks: list[Chunk] = []

    with CHUNKS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            record = json.loads(line)

            chunks.append(
                Chunk(
                    id=record["id"],
                    document_id=(
                        record["document_id"]
                    ),
                    chunk_index=(
                        record["chunk_index"]
                    ),
                    text=record["text"],
                    start_char=(
                        record["start_char"]
                    ),
                    end_char=(
                        record["end_char"]
                    ),
                    metadata=(
                        record["metadata"]
                    ),
                )
            )

    return chunks


def main() -> None:

    print("=" * 90)
    print(
        "PHASE 9.5D — PERSISTED "
        "DENSE RETRIEVAL REPRODUCTION TEST"
    )
    print("=" * 90)

    # ---------------------------------------------------------
    # Load persisted corpus artifacts
    # ---------------------------------------------------------

    print("\nLoading frozen chunks...")

    chunks = load_chunks()

    print(
        f"Chunks: {len(chunks):,}"
    )

    assert (
        len(chunks)
        == EXPECTED_CHUNKS
    )

    print(
        "Loading persisted embeddings..."
    )

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    assert embeddings.shape == (
        EXPECTED_CHUNKS,
        EXPECTED_DIMENSION,
    )

    assert (
        embeddings.dtype
        == np.float32
    )

    print(
        "Loading persisted FAISS index..."
    )

    persisted_index = (
        faiss.read_index(
            str(FAISS_INDEX_PATH)
        )
    )

    assert (
        persisted_index.ntotal
        == EXPECTED_CHUNKS
    )

    assert (
        persisted_index.d
        == EXPECTED_DIMENSION
    )

    # ---------------------------------------------------------
    # Reconstruct existing DenseRetriever
    #
    # DenseRetriever should build/use its normal exact-IP index
    # over the same frozen embedding matrix.
    # ---------------------------------------------------------

    print(
        "\nConstructing DenseRetriever "
        "from persisted embeddings..."
    )

    dense_retriever = DenseRetriever(
        chunks=chunks,
        embeddings=embeddings,
    )

    # ---------------------------------------------------------
    # Load one real HotpotQA query
    #
    # Use the first example from our historical exploratory
    # slice so the query itself is familiar and deterministic.
    # ---------------------------------------------------------

    print("\nLoading test query...")

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    raw_example = dataset["train"][0]

    query = raw_example["question"]

    print(
        f"Query: {query}"
    )

    # ---------------------------------------------------------
    # Load the same MiniLM query encoder
    # ---------------------------------------------------------

    print(
        "\nLoading dense embedder..."
    )

    embedder = DenseEmbedder(
        DenseEmbeddingConfig(
            model_name=MODEL_NAME,
            batch_size=32,
            normalize_embeddings=True,
        )
    )

    query_embedding = (
        embedder.encode_query(
            query
        )
    )

    assert query_embedding.shape == (
        EXPECTED_DIMENSION,
    )

    assert np.isclose(
        np.linalg.norm(
            query_embedding
        ),
        1.0,
        atol=1e-5,
    )

    # ---------------------------------------------------------
    # Direct persisted-FAISS search
    # ---------------------------------------------------------

    direct_scores, direct_indices = (
        persisted_index.search(
            query_embedding.reshape(
                1,
                -1,
            ),
            TOP_K,
        )
    )

    direct_indices_list = [
        int(index)
        for index
        in direct_indices[0]
    ]

    direct_scores_list = [
        float(score)
        for score
        in direct_scores[0]
    ]

    direct_chunk_ids = [
        chunks[index].id
        for index
        in direct_indices_list
    ]

    # ---------------------------------------------------------
    # Existing DenseRetriever search
    # ---------------------------------------------------------

    retriever_results = (
        dense_retriever.search(
            query=query,
            embedder=embedder,
            top_k=TOP_K,
        )
    )

    retriever_chunk_ids = [
        result.chunk.id
        for result
        in retriever_results
    ]

    retriever_scores = [
        float(result.score)
        for result
        in retriever_results
    ]

    # ---------------------------------------------------------
    # Compare
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print("TOP-10 COMPARISON")
    print("=" * 90)

    for rank in range(TOP_K):

        direct_id = (
            direct_chunk_ids[rank]
        )

        retriever_id = (
            retriever_chunk_ids[rank]
        )

        direct_score = (
            direct_scores_list[rank]
        )

        retriever_score = (
            retriever_scores[rank]
        )

        print(
            f"\nRank {rank + 1}"
        )

        print(
            f"  Direct FAISS: "
            f"{direct_id}"
        )

        print(
            f"  DenseRetriever: "
            f"{retriever_id}"
        )

        print(
            f"  Direct score: "
            f"{direct_score:.6f}"
        )

        print(
            f"  Retriever score: "
            f"{retriever_score:.6f}"
        )

    # ---------------------------------------------------------
    # Exact ranking + numerical score agreement
    # ---------------------------------------------------------

    assert (
        direct_chunk_ids
        == retriever_chunk_ids
    ), (
        "Persisted FAISS ranking and "
        "DenseRetriever ranking differ."
    )

    assert np.allclose(
        np.asarray(
            direct_scores_list,
            dtype=np.float32,
        ),
        np.asarray(
            retriever_scores,
            dtype=np.float32,
        ),
        atol=1e-6,
    ), (
        "Persisted FAISS scores and "
        "DenseRetriever scores differ."
    )

    # ---------------------------------------------------------
    # Basic result integrity
    # ---------------------------------------------------------

    assert (
        len(retriever_results)
        == TOP_K
    )

    assert (
        len(set(retriever_chunk_ids))
        == TOP_K
    )

    print("\n" + "=" * 90)

    print(
        "PASS: Persisted FAISS search and "
        "DenseRetriever reproduce the same "
        "top-10 ranking and scores."
    )

    print(
        "PASS: Frozen chunk rows, embedding "
        "rows, and retrieval metadata remain "
        "aligned."
    )


if __name__ == "__main__":
    main()