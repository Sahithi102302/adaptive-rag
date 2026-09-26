from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class DenseEmbeddingConfig:
    """
    Configuration for the dense embedding model.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    normalize_embeddings: bool = True

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError(
                "model_name must be a non-empty string."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0."
            )


class DenseEmbedder:
    """
    Thin wrapper around SentenceTransformer for producing dense
    document/chunk and query embeddings.

    The wrapper gives AdaptiveRAG one stable embedding interface so
    the retrieval system does not depend directly on a particular
    embedding model implementation.
    """

    def __init__(
        self,
        config: DenseEmbeddingConfig | None = None,
    ) -> None:
        self.config = config or DenseEmbeddingConfig()

        self.model = SentenceTransformer(
            self.config.model_name
        )

        self.embedding_dimension = (
            self.model.get_embedding_dimension()
        )

        if self.embedding_dimension is None:
            raise RuntimeError(
                "Could not determine embedding dimension."
            )

    def encode(
        self,
        texts: Sequence[str],
        *,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Encode one or more texts into float32 dense vectors.

        Returns
        -------
        np.ndarray
            Matrix with shape:

                (number_of_texts, embedding_dimension)
        """

        if not texts:
            return np.empty(
                (
                    0,
                    self.embedding_dimension,
                ),
                dtype=np.float32,
            )

        cleaned_texts: list[str] = []

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"Text at position {index} "
                    "must be a string."
                )

            cleaned = text.strip()

            if not cleaned:
                raise ValueError(
                    f"Text at position {index} "
                    "cannot be empty."
                )

            cleaned_texts.append(cleaned)

        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=self.config.batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=(
                self.config.normalize_embeddings
            ),
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise RuntimeError(
                "Embedding model returned an unexpected "
                f"shape: {embeddings.shape}"
            )

        if (
            embeddings.shape[1]
            != self.embedding_dimension
        ):
            raise RuntimeError(
                "Embedding dimension mismatch. "
                f"Expected {self.embedding_dimension}, "
                f"received {embeddings.shape[1]}."
            )

        return embeddings

    def encode_query(
        self,
        query: str,
    ) -> np.ndarray:
        """
        Encode a single query.

        Returns a one-dimensional embedding vector.
        """

        embeddings = self.encode([query])

        return embeddings[0]

    def encode_documents(
        self,
        texts: Sequence[str],
        *,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Encode document or chunk texts.
        """

        return self.encode(
            texts,
            show_progress_bar=show_progress_bar,
        )