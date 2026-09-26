from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

from src.chunking.chunk import Chunk


_TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """
    Simple deterministic lexical tokenizer for the BM25 baseline.

    Lowercases text and extracts word-like tokens.

    This intentionally avoids stemming, stop-word removal, and other
    preprocessing so the first sparse baseline remains simple and
    reproducible.
    """

    return _TOKEN_PATTERN.findall(
        text.lower()
    )


@dataclass(frozen=True)
class BM25Config:
    """
    Standard BM25 hyperparameters.

    k1 controls term-frequency saturation.
    b controls document-length normalization.
    """

    k1: float = 1.5
    b: float = 0.75


@dataclass(frozen=True)
class SparseRetrievalResult:
    """
    One result returned by sparse BM25 retrieval.
    """

    rank: int
    score: float
    chunk: Chunk


class BM25Retriever:
    """
    Exact in-memory BM25 retriever over chunk representations.

    This implementation is intended as a transparent sparse
    retrieval baseline before hybrid dense+sparse fusion.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        retrieval_texts: Sequence[str],
        config: BM25Config | None = None,
    ) -> None:

        self.chunks = list(chunks)
        self.retrieval_texts = list(
            retrieval_texts
        )

        self.config = (
            config
            if config is not None
            else BM25Config()
        )

        if not self.chunks:
            raise ValueError(
                "Cannot build BM25 index with "
                "zero chunks."
            )

        if (
            len(self.chunks)
            != len(self.retrieval_texts)
        ):
            raise ValueError(
                "Number of chunks must match number "
                "of retrieval texts."
            )

        if self.config.k1 <= 0:
            raise ValueError(
                "BM25 k1 must be greater than 0."
            )

        if not (
            0.0 <= self.config.b <= 1.0
        ):
            raise ValueError(
                "BM25 b must be between 0 and 1."
            )

        self._document_term_frequencies: list[
            Counter[str]
        ] = []

        self._document_lengths: list[int] = []

        document_frequency: dict[
            str,
            int
        ] = defaultdict(int)

        # -----------------------------------------------------
        # Tokenize corpus and compute term statistics
        # -----------------------------------------------------

        for text in self.retrieval_texts:

            tokens = tokenize(text)

            term_frequencies = Counter(
                tokens
            )

            self._document_term_frequencies.append(
                term_frequencies
            )

            self._document_lengths.append(
                len(tokens)
            )

            for term in term_frequencies:
                document_frequency[term] += 1

        self._document_count = len(
            self.chunks
        )

        self._average_document_length = (
            sum(self._document_lengths)
            / self._document_count
        )

        # -----------------------------------------------------
        # Precompute BM25 inverse document frequencies
        # -----------------------------------------------------

        self._idf: dict[
            str,
            float
        ] = {}

        for (
            term,
            frequency,
        ) in document_frequency.items():

            self._idf[term] = math.log(
                1.0
                + (
                    self._document_count
                    - frequency
                    + 0.5
                )
                / (
                    frequency
                    + 0.5
                )
            )

    @property
    def size(self) -> int:
        return self._document_count

    @property
    def average_document_length(
        self,
    ) -> float:
        return (
            self._average_document_length
        )

    def _score_document(
        self,
        query_terms: Counter[str],
        document_index: int,
    ) -> float:
        """
        Compute the BM25 score for one indexed chunk.
        """

        document_tf = (
            self._document_term_frequencies[
                document_index
            ]
        )

        document_length = (
            self._document_lengths[
                document_index
            ]
        )

        score = 0.0

        k1 = self.config.k1
        b = self.config.b

        for term in query_terms:

            term_frequency = (
                document_tf.get(
                    term,
                    0,
                )
            )

            if term_frequency == 0:
                continue

            idf = self._idf.get(
                term
            )

            if idf is None:
                continue

            denominator = (
                term_frequency
                + k1
                * (
                    1.0
                    - b
                    + b
                    * (
                        document_length
                        / self._average_document_length
                    )
                )
            )

            score += (
                idf
                * (
                    term_frequency
                    * (k1 + 1.0)
                )
                / denominator
            )

        return float(score)

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[SparseRetrievalResult]:
        """
        Retrieve the highest-scoring chunks for a text query.
        """

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        query_tokens = tokenize(
            query
        )

        if not query_tokens:
            return []

        query_terms = Counter(
            query_tokens
        )

        scored_documents: list[
            tuple[int, float]
        ] = []

        for document_index in range(
            self._document_count
        ):

            score = self._score_document(
                query_terms=query_terms,
                document_index=document_index,
            )

            if score > 0.0:
                scored_documents.append(
                    (
                        document_index,
                        score,
                    )
                )

        # Deterministic tie breaking:
        # higher score first, then lower corpus index.
        scored_documents.sort(
            key=lambda item: (
                -item[1],
                item[0],
            )
        )

        actual_k = min(
            top_k,
            len(scored_documents),
        )

        results: list[
            SparseRetrievalResult
        ] = []

        for rank, (
            document_index,
            score,
        ) in enumerate(
            scored_documents[:actual_k],
            start=1,
        ):

            results.append(
                SparseRetrievalResult(
                    rank=rank,
                    score=score,
                    chunk=self.chunks[
                        document_index
                    ],
                )
            )

        return results