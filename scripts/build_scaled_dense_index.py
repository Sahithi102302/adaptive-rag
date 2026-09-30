from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src.chunking.chunk import Chunk
from src.embeddings.dense_embedder import (
    DenseEmbedder,
    DenseEmbeddingConfig,
)
from src.retrieval.dense_retriever import (
    format_chunk_for_retrieval,
)


INPUT_DIR = Path(
    "artifacts/diagnosis/scaled"
)

CHUNKS_PATH = (
    INPUT_DIR
    / "pooled_chunks.jsonl"
)

EMBEDDINGS_PATH = (
    INPUT_DIR
    / "pooled_embeddings.npy"
)

FAISS_INDEX_PATH = (
    INPUT_DIR
    / "pooled_faiss.index"
)

INDEX_MANIFEST_PATH = (
    INPUT_DIR
    / "dense_index_manifest.json"
)


MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

EXPECTED_CHUNKS = 32368
EXPECTED_DIMENSION = 384


def load_chunks() -> list[Chunk]:
    """
    Load the exact frozen chunk corpus created in Phase 9.5B.

    JSONL order is preserved because FAISS row i must correspond
    exactly to chunks[i].
    """

    chunks: list[Chunk] = []

    with CHUNKS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            record = json.loads(
                line
            )

            chunk = Chunk(
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

            chunks.append(chunk)

    return chunks


def main() -> None:

    print("=" * 90)
    print(
        "PHASE 9.5C — BUILD PERSISTENT "
        "DENSE EMBEDDINGS + FAISS INDEX"
    )
    print("=" * 90)

    # ---------------------------------------------------------
    # Load frozen chunks
    # ---------------------------------------------------------

    print(
        "\nLoading frozen chunk corpus..."
    )

    chunks = load_chunks()

    print(
        f"Chunks loaded: "
        f"{len(chunks):,}"
    )

    assert (
        len(chunks)
        == EXPECTED_CHUNKS
    )

    chunk_ids = [
        chunk.id
        for chunk in chunks
    ]

    assert (
        len(chunk_ids)
        == len(set(chunk_ids))
    ), "Duplicate chunk IDs detected."

    # ---------------------------------------------------------
    # Build exact retrieval representation
    #
    # This MUST match the representation used in our previous
    # dense-retrieval experiments: title + chunk text.
    # ---------------------------------------------------------

    print(
        "\nPreparing retrieval texts..."
    )

    retrieval_texts = [
        format_chunk_for_retrieval(
            chunk
        )
        for chunk in chunks
    ]

    assert (
        len(retrieval_texts)
        == len(chunks)
    )

    # ---------------------------------------------------------
    # Load MiniLM
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

    assert (
        embedder.embedding_dimension
        == EXPECTED_DIMENSION
    )

    # ---------------------------------------------------------
    # Embed frozen corpus
    # ---------------------------------------------------------

    print(
        f"\nEmbedding "
        f"{len(retrieval_texts):,} chunks..."
    )

    embeddings = (
        embedder.encode_documents(
            retrieval_texts,
            show_progress_bar=True,
        )
    )

    # ---------------------------------------------------------
    # Embedding validation
    # ---------------------------------------------------------

    print(
        "\nValidating embeddings..."
    )

    assert embeddings.shape == (
        EXPECTED_CHUNKS,
        EXPECTED_DIMENSION,
    )

    assert (
        embeddings.dtype
        == np.float32
    )

    assert np.isfinite(
        embeddings
    ).all()

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    min_norm = float(
        np.min(norms)
    )

    max_norm = float(
        np.max(norms)
    )

    mean_norm = float(
        np.mean(norms)
    )

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    print(
        f"Embedding dtype: "
        f"{embeddings.dtype}"
    )

    print(
        f"Norm min:  "
        f"{min_norm:.6f}"
    )

    print(
        f"Norm mean: "
        f"{mean_norm:.6f}"
    )

    print(
        f"Norm max:  "
        f"{max_norm:.6f}"
    )

    assert np.allclose(
        norms,
        1.0,
        atol=1e-5,
    ), (
        "Embeddings were expected to be "
        "L2-normalized."
    )

    # ---------------------------------------------------------
    # Persist embeddings
    # ---------------------------------------------------------

    print(
        f"\nSaving embeddings to "
        f"{EMBEDDINGS_PATH}..."
    )

    np.save(
        EMBEDDINGS_PATH,
        embeddings,
    )

    # ---------------------------------------------------------
    # Build exact FAISS index
    #
    # Since vectors are normalized:
    #
    # inner product == cosine similarity
    # ---------------------------------------------------------

    print(
        "\nBuilding FAISS IndexFlatIP..."
    )

    index = faiss.IndexFlatIP(
        EXPECTED_DIMENSION
    )

    index.add(
        embeddings
    )

    assert (
        index.ntotal
        == EXPECTED_CHUNKS
    )

    print(
        f"FAISS vectors: "
        f"{index.ntotal:,}"
    )

    # ---------------------------------------------------------
    # Persist FAISS index
    # ---------------------------------------------------------

    print(
        f"\nSaving FAISS index to "
        f"{FAISS_INDEX_PATH}..."
    )

    faiss.write_index(
        index,
        str(
            FAISS_INDEX_PATH
        ),
    )

    # ---------------------------------------------------------
    # Reload artifacts and verify
    # ---------------------------------------------------------

    print(
        "\nReloading persisted artifacts..."
    )

    reloaded_embeddings = np.load(
        EMBEDDINGS_PATH
    )

    reloaded_index = (
        faiss.read_index(
            str(
                FAISS_INDEX_PATH
            )
        )
    )

    assert (
        reloaded_embeddings.shape
        == embeddings.shape
    )

    assert (
        reloaded_embeddings.dtype
        == np.float32
    )

    assert (
        reloaded_index.ntotal
        == EXPECTED_CHUNKS
    )

    assert (
        reloaded_index.d
        == EXPECTED_DIMENSION
    )

    # Exact .npy round trip.
    assert np.array_equal(
        embeddings,
        reloaded_embeddings,
    )

    # ---------------------------------------------------------
    # Retrieval smoke test
    #
    # Search using the first corpus vector itself.
    # Its own row should have cosine similarity ~1 and should
    # appear at rank 1.
    # ---------------------------------------------------------

    print(
        "\nRunning persisted-index "
        "smoke test..."
    )

    query_vector = (
        reloaded_embeddings[
            0:1
        ]
    )

    scores, indices = (
        reloaded_index.search(
            query_vector,
            5,
        )
    )

    top_index = int(
        indices[0][0]
    )

    top_score = float(
        scores[0][0]
    )

    print(
        f"Top row index: "
        f"{top_index}"
    )

    print(
        f"Top similarity: "
        f"{top_score:.6f}"
    )

    print(
        f"Top chunk ID: "
        f"{chunks[top_index].id}"
    )

    assert (
        top_index == 0
    ), (
        "FAISS row/chunk alignment "
        "smoke test failed."
    )

    assert np.isclose(
        top_score,
        1.0,
        atol=1e-5,
    )

    # ---------------------------------------------------------
    # Save dense-index manifest
    # ---------------------------------------------------------

    manifest = {
        "embedding_model": (
            MODEL_NAME
        ),

        "embedding_dimension": (
            EXPECTED_DIMENSION
        ),

        "embedding_dtype": (
            "float32"
        ),

        "normalized_embeddings": True,

        "retrieval_representation": (
            "format_chunk_for_retrieval: "
            "title + chunk text"
        ),

        "faiss_index_type": (
            "IndexFlatIP"
        ),

        "similarity": (
            "cosine via inner product "
            "over L2-normalized vectors"
        ),

        "chunk_count": (
            EXPECTED_CHUNKS
        ),

        "chunk_artifact": (
            str(CHUNKS_PATH)
        ),

        "embedding_artifact": (
            str(EMBEDDINGS_PATH)
        ),

        "faiss_artifact": (
            str(FAISS_INDEX_PATH)
        ),

        "row_alignment": (
            "FAISS row i corresponds to "
            "JSONL chunk row i"
        ),
    }

    with INDEX_MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------

    embedding_size_mib = (
        EMBEDDINGS_PATH.stat().st_size
        / (1024 ** 2)
    )

    index_size_mib = (
        FAISS_INDEX_PATH.stat().st_size
        / (1024 ** 2)
    )

    print("\n" + "=" * 90)
    print("DENSE INDEX SUMMARY")
    print("=" * 90)

    print(
        f"\nChunks: "
        f"{EXPECTED_CHUNKS:,}"
    )

    print(
        f"Embedding dimension: "
        f"{EXPECTED_DIMENSION}"
    )

    print(
        f"Embedding matrix: "
        f"{embedding_size_mib:.2f} MiB"
    )

    print(
        f"FAISS index: "
        f"{index_size_mib:.2f} MiB"
    )

    print(
        f"\nEmbeddings: "
        f"{EMBEDDINGS_PATH}"
    )

    print(
        f"FAISS index: "
        f"{FAISS_INDEX_PATH}"
    )

    print(
        f"Manifest: "
        f"{INDEX_MANIFEST_PATH}"
    )

    print(
        "\nPASS: Persistent dense "
        "retrieval artifacts built "
        "and verified."
    )


if __name__ == "__main__":
    main()