from dataclasses import dataclass
from typing import Sequence

import faiss
import numpy as np

from src.chunking.chunk import Chunk
from src.embeddings.dense_embedder import DenseEmbedder


@dataclass(frozen=True)
class DenseRetrievalResult:
    """
    One result returned by dense retrieval.
    """

    rank: int
    score: float
    chunk: Chunk


def format_chunk_for_retrieval(
    chunk: Chunk,
) -> str:
    """
    Construct the text representation used for dense retrieval.

    The original Chunk.text remains unchanged. We prepend the source
    title only for embedding/search because titles contain useful
    document-level identity information.
    """

    title = chunk.metadata.get("title")

    if isinstance(title, str) and title.strip():
        return f"{title.strip()}: {chunk.text}"

    return chunk.text


class DenseRetriever:
    """
    Exact dense retriever using FAISS IndexFlatIP.

    Because embeddings are L2-normalized, inner product corresponds
    to cosine similarity.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embeddings: np.ndarray,
    ) -> None:
        self.chunks = list(chunks)

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D matrix."
            )

        if len(self.chunks) != embeddings.shape[0]:
            raise ValueError(
                "Number of chunks must match number "
                "of embedding rows. "
                f"Received {len(self.chunks)} chunks "
                f"and {embeddings.shape[0]} vectors."
            )

        if embeddings.shape[0] == 0:
            raise ValueError(
                "Cannot build a dense index with "
                "zero embeddings."
            )

        self.embedding_dimension = (
            embeddings.shape[1]
        )

        self.index = faiss.IndexFlatIP(
            self.embedding_dimension
        )

        self.index.add(
            np.ascontiguousarray(
                embeddings,
                dtype=np.float32,
            )
        )

        if self.index.ntotal != len(self.chunks):
            raise RuntimeError(
                "FAISS index size does not match "
                "chunk count."
            )

    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    def search_by_vector(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[DenseRetrievalResult]:
        """
        Search the index using an already-computed query vector.
        """

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_embedding.ndim == 1:
            query_embedding = (
                query_embedding.reshape(1, -1)
            )

        if (
            query_embedding.ndim != 2
            or query_embedding.shape[0] != 1
        ):
            raise ValueError(
                "query_embedding must represent "
                "exactly one query."
            )

        if (
            query_embedding.shape[1]
            != self.embedding_dimension
        ):
            raise ValueError(
                "Query embedding dimension does not "
                "match index dimension."
            )

        actual_k = min(
            top_k,
            self.size,
        )

        scores, indices = self.index.search(
            np.ascontiguousarray(
                query_embedding,
                dtype=np.float32,
            ),
            actual_k,
        )

        results: list[DenseRetrievalResult] = []

        for rank, (
            vector_index,
            score,
        ) in enumerate(
            zip(
                indices[0],
                scores[0],
            ),
            start=1,
        ):
            if vector_index < 0:
                continue

            results.append(
                DenseRetrievalResult(
                    rank=rank,
                    score=float(score),
                    chunk=self.chunks[
                        int(vector_index)
                    ],
                )
            )

        return results

    def search(
        self,
        query: str,
        embedder: DenseEmbedder,
        top_k: int = 10,
    ) -> list[DenseRetrievalResult]:
        """
        Embed a query and retrieve its nearest chunks.
        """

        query_embedding = (
            embedder.encode_query(query)
        )

        return self.search_by_vector(
            query_embedding=query_embedding,
            top_k=top_k,
        )