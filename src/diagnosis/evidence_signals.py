from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.embeddings.dense_embedder import DenseEmbedder
from src.retrieval.reranker import RerankedRetrievalResult


_TOKEN_PATTERN = re.compile(
    r"\b\w+\b",
    re.UNICODE,
)

# Small deterministic stopword set.
#
# The goal is not linguistic perfection. We want an interpretable
# approximation of the informative lexical content of the query.
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "with",
}


@dataclass(frozen=True)
class LexicalEvidenceSignals:
    """
    Runtime lexical signals describing how the retrieved evidence
    covers the informative tokens appearing in the query.

    These signals use only:
        - the query
        - retrieved chunk text

    They never use gold supporting facts or answer labels.
    """

    query_token_count: int
    informative_query_token_count: int

    covered_query_token_count: int
    query_token_coverage: float | None

    repeated_query_token_count: int
    repeated_query_token_fraction: float | None

    mean_query_token_document_frequency: float | None
    min_query_token_document_frequency: int | None
    max_query_token_document_frequency: int | None

    uncovered_query_token_count: int


@dataclass(frozen=True)
class SemanticEvidenceSignals:
    """
    Runtime semantic signals describing relationships between
    the query and the retrieved evidence set.

    These signals use only:
        - the query
        - retrieved chunk text
        - the runtime embedding model

    They never use gold supporting facts, answers, or oracle
    diagnoses.

    When embeddings are L2-normalized, the dot products used
    below are cosine similarities.
    """

    max_query_chunk_similarity: float | None
    mean_query_chunk_similarity: float | None
    semantic_coverage_spread: float | None
    mean_pairwise_chunk_similarity: float | None


def tokenize(
    text: str,
) -> list[str]:
    """
    Deterministic lowercase lexical tokenizer.
    """

    return [
        token.lower()
        for token in _TOKEN_PATTERN.findall(
            text
        )
    ]


def informative_query_tokens(
    query: str,
) -> list[str]:
    """
    Return unique informative query tokens while preserving
    first-occurrence order.
    """

    tokens = tokenize(query)

    informative = [
        token
        for token in tokens
        if token not in _STOPWORDS
    ]

    # Deduplicate while preserving first-occurrence order.
    return list(
        dict.fromkeys(
            informative
        )
    )


def extract_lexical_evidence_signals(
    query: str,
    results: Sequence[
        RerankedRetrievalResult
    ],
) -> LexicalEvidenceSignals:
    """
    Extract lexical evidence-coverage signals.

    Only the query and retrieved chunk text are used.
    No gold evidence or oracle labels are required.
    """

    all_query_tokens = tokenize(
        query
    )

    query_tokens = (
        informative_query_tokens(
            query
        )
    )

    informative_count = len(
        query_tokens
    )

    # ---------------------------------------------------------
    # Empty informative query
    # ---------------------------------------------------------

    if informative_count == 0:
        return LexicalEvidenceSignals(
            query_token_count=len(
                all_query_tokens
            ),
            informative_query_token_count=0,
            covered_query_token_count=0,
            query_token_coverage=None,
            repeated_query_token_count=0,
            repeated_query_token_fraction=None,
            mean_query_token_document_frequency=None,
            min_query_token_document_frequency=None,
            max_query_token_document_frequency=None,
            uncovered_query_token_count=0,
        )

    # ---------------------------------------------------------
    # Build document-level token sets
    #
    # Multiple chunks from the same source document are merged
    # conceptually for document-frequency calculations.
    # ---------------------------------------------------------

    document_tokens: dict[
        str,
        set[str],
    ] = {}

    for result in results:

        document_id = (
            result.chunk.document_id
        )

        tokens = set(
            tokenize(
                result.chunk.text
            )
        )

        if (
            document_id
            not in document_tokens
        ):
            document_tokens[
                document_id
            ] = set()

        document_tokens[
            document_id
        ].update(
            tokens
        )

    # ---------------------------------------------------------
    # Query-token document frequency
    # ---------------------------------------------------------

    token_document_frequency: Counter[
        str
    ] = Counter()

    for token in query_tokens:

        for tokens in (
            document_tokens.values()
        ):

            if token in tokens:
                token_document_frequency[
                    token
                ] += 1

    covered_tokens = [
        token
        for token in query_tokens
        if (
            token_document_frequency[
                token
            ]
            > 0
        )
    ]

    uncovered_tokens = [
        token
        for token in query_tokens
        if (
            token_document_frequency[
                token
            ]
            == 0
        )
    ]

    repeated_tokens = [
        token
        for token in query_tokens
        if (
            token_document_frequency[
                token
            ]
            >= 2
        )
    ]

    frequencies = [
        token_document_frequency[
            token
        ]
        for token in query_tokens
    ]

    # ---------------------------------------------------------
    # Aggregate lexical signals
    # ---------------------------------------------------------

    covered_count = len(
        covered_tokens
    )

    repeated_count = len(
        repeated_tokens
    )

    query_token_coverage = (
        covered_count
        / informative_count
    )

    repeated_fraction = (
        repeated_count
        / informative_count
    )

    mean_document_frequency = (
        sum(frequencies)
        / informative_count
    )

    return LexicalEvidenceSignals(
        query_token_count=len(
            all_query_tokens
        ),
        informative_query_token_count=(
            informative_count
        ),
        covered_query_token_count=(
            covered_count
        ),
        query_token_coverage=(
            query_token_coverage
        ),
        repeated_query_token_count=(
            repeated_count
        ),
        repeated_query_token_fraction=(
            repeated_fraction
        ),
        mean_query_token_document_frequency=(
            mean_document_frequency
        ),
        min_query_token_document_frequency=(
            min(frequencies)
        ),
        max_query_token_document_frequency=(
            max(frequencies)
        ),
        uncovered_query_token_count=(
            len(uncovered_tokens)
        ),
    )


def extract_semantic_evidence_signals(
    query: str,
    results: Sequence[
        RerankedRetrievalResult
    ],
    embedder: DenseEmbedder,
) -> SemanticEvidenceSignals:
    """
    Extract semantic evidence-set signals using the existing
    dense embedding model.

    The function uses only inference-time information:
        - query
        - retrieved chunk text
        - embedding model

    It does not use:
        - HotpotQA supporting facts
        - gold document IDs
        - gold sentence IDs
        - oracle failure labels
        - reference answers

    If DenseEmbedder is configured with normalized embeddings,
    dot products below are cosine similarities.
    """

    # ---------------------------------------------------------
    # Empty retrieval
    # ---------------------------------------------------------

    if not results:
        return SemanticEvidenceSignals(
            max_query_chunk_similarity=None,
            mean_query_chunk_similarity=None,
            semantic_coverage_spread=None,
            mean_pairwise_chunk_similarity=None,
        )

    # ---------------------------------------------------------
    # Encode query
    # ---------------------------------------------------------

    query_embedding = (
        embedder.encode_query(
            query
        )
    )

    # ---------------------------------------------------------
    # Encode retrieved evidence
    #
    # Deliberately use raw chunk text rather than the retrieval
    # representation containing title + chunk text.
    #
    # This first semantic experiment measures evidence passage
    # content without allowing titles to artificially increase
    # query similarity.
    # ---------------------------------------------------------

    chunk_texts = [
        result.chunk.text
        for result in results
    ]

    chunk_embeddings = (
        embedder.encode_documents(
            chunk_texts
        )
    )

    # ---------------------------------------------------------
    # Query -> chunk semantic similarities
    #
    # Shapes:
    #
    # chunk_embeddings:
    #     (number_of_chunks, embedding_dimension)
    #
    # query_embedding:
    #     (embedding_dimension,)
    #
    # result:
    #     (number_of_chunks,)
    # ---------------------------------------------------------

    query_chunk_similarities = (
        chunk_embeddings
        @ query_embedding
    )

    max_query_chunk_similarity = float(
        np.max(
            query_chunk_similarities
        )
    )

    mean_query_chunk_similarity = float(
        np.mean(
            query_chunk_similarities
        )
    )

    # ---------------------------------------------------------
    # Semantic spread
    #
    # Large values mean the best-matching chunk is much more
    # similar to the query than the retrieved set on average.
    #
    # This is only a diagnostic signal, not a sufficiency rule.
    # ---------------------------------------------------------

    semantic_coverage_spread = (
        max_query_chunk_similarity
        - mean_query_chunk_similarity
    )

    # ---------------------------------------------------------
    # Chunk -> chunk semantic similarity
    #
    # We average only unique unordered chunk pairs:
    #
    # (0,1), (0,2), ..., (1,2), ...
    #
    # Self-similarities on the diagonal are excluded.
    #
    # A higher mean may indicate a more semantically redundant
    # evidence set, but that interpretation is a hypothesis to
    # test rather than a hard-coded rule.
    # ---------------------------------------------------------

    if len(results) < 2:

        mean_pairwise_chunk_similarity = (
            None
        )

    else:

        similarity_matrix = (
            chunk_embeddings
            @ chunk_embeddings.T
        )

        upper_triangle = (
            np.triu_indices(
                len(results),
                k=1,
            )
        )

        pairwise_similarities = (
            similarity_matrix[
                upper_triangle
            ]
        )

        mean_pairwise_chunk_similarity = (
            float(
                np.mean(
                    pairwise_similarities
                )
            )
        )

    return SemanticEvidenceSignals(
        max_query_chunk_similarity=(
            max_query_chunk_similarity
        ),
        mean_query_chunk_similarity=(
            mean_query_chunk_similarity
        ),
        semantic_coverage_spread=(
            semantic_coverage_spread
        ),
        mean_pairwise_chunk_similarity=(
            mean_pairwise_chunk_similarity
        ),
    )