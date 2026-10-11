from __future__ import annotations

import numpy as np

from src.diagnosis.oracle import (
    diagnose_oracle_retrieval,
)
from src.diagnosis.states import FailureType
from src.embeddings.dense_embedder import (
    DenseEmbedder,
    DenseEmbeddingConfig,
)
from src.recovery.multi_query import (
    MultiQueryRecovery,
    MultiQueryRecoveryConfig,
)
from src.recovery.states import (
    RecoveryAction,
    RecoveryRequest,
)
from src.retrieval.dense_retriever import (
    DenseRetriever,
    format_chunk_for_retrieval,
)
from src.retrieval.reranker import (
    CrossEncoderReranker,
    RerankerConfig,
)
from src.retrieval.sparse_retriever import (
    BM25Retriever,
)

from scripts.build_scaled_diagnostic_dataset import (
    CROSS_ENCODER_MODEL_NAME,
    DENSE_MODEL_NAME,
    EMBEDDINGS_PATH,
    EXPECTED_CHUNKS,
    EXPECTED_DIMENSION,
    get_gold_evidence,
    load_frozen_chunks,
    load_query_examples,
)

from scripts.evaluate_expanded_candidate_recovery import (
    get_retrieved_evidence,
)


# ============================================================
# Configuration
# ============================================================

SPLIT_NAME = "validation"

BASELINE_CANDIDATE_K = 20
FINAL_K = 10

BATCH_SIZE = 32

MAX_CASES = 10

MULTI_QUERY_CONFIG = (
    MultiQueryRecoveryConfig(
        max_bridge_titles=3,
        source_result_depth=5,
    )
)


# ============================================================
# Main
# ============================================================


def main() -> None:

    print("=" * 100)
    print(
        "PHASE 10.3B — REAL RESIDUAL "
        "MULTI-QUERY INSPECTION"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # Frozen validation queries
    # --------------------------------------------------------

    print(
        "\nLoading validation queries..."
    )

    examples = load_query_examples(
        SPLIT_NAME
    )

    assert len(examples) == 200

    # --------------------------------------------------------
    # Frozen corpus
    # --------------------------------------------------------

    print(
        "Loading frozen chunks..."
    )

    chunks = load_frozen_chunks()

    assert len(chunks) == EXPECTED_CHUNKS

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

    assert embeddings.dtype == np.float32

    retrieval_texts = [
        format_chunk_for_retrieval(
            chunk
        )
        for chunk in chunks
    ]

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    print(
        "Constructing DenseRetriever..."
    )

    dense_retriever = DenseRetriever(
        chunks=chunks,
        embeddings=embeddings,
    )

    print(
        "Building BM25 index..."
    )

    sparse_retriever = BM25Retriever(
        chunks=chunks,
        retrieval_texts=retrieval_texts,
    )

    print(
        "Loading dense embedder..."
    )

    embedder = DenseEmbedder(
        DenseEmbeddingConfig(
            model_name=DENSE_MODEL_NAME,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
        )
    )

    print(
        "Loading cross-encoder..."
    )

    reranker = CrossEncoderReranker(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        config=RerankerConfig(
            model_name=(
                CROSS_ENCODER_MODEL_NAME
            ),
            candidate_k=(
                BASELINE_CANDIDATE_K
            ),
            batch_size=BATCH_SIZE,
        ),
    )

    recovery = MultiQueryRecovery(
        MULTI_QUERY_CONFIG
    )

    # --------------------------------------------------------
    # Find real baseline failures
    # --------------------------------------------------------

    inspected = 0

    print(
        "\nSearching for residual-style "
        "baseline failures...\n"
    )

    for position, example in enumerate(
        examples,
        start=1,
    ):

        gold = get_gold_evidence(
            example
        )

        if not gold:
            continue

        results = reranker.search(
            query=example.question,
            embedder=embedder,
            top_k=FINAL_K,
        )

        retrieved_evidence = (
            get_retrieved_evidence(
                results
            )
        )

        if gold.issubset(
            retrieved_evidence
        ):
            continue

        diagnosis = (
            diagnose_oracle_retrieval(
                gold_evidence=gold,
                results=results,
            )
        )

        # Focus first on the dominant mechanism
        # from the validation failures.
        if (
            diagnosis.failure_type
            != FailureType
            .EVIDENCE_COVERAGE_FAILURE
        ):
            continue

        request = RecoveryRequest(
            original_query=(
                example.question
            ),
            action=(
                RecoveryAction
                .MULTI_QUERY_RETRIEVAL
            ),
            original_results=tuple(
                results
            ),
            metadata={},
        )

        recovery_result = (
            recovery.recover(
                request
            )
        )

        missing = (
            gold
            - retrieved_evidence
        )

        inspected += 1

        print(
            "=" * 100
        )

        print(
            f"CASE {inspected}"
        )

        print(
            "=" * 100
        )

        print(
            f"\nValidation position: "
            f"{position}"
        )

        print(
            f"\nQuestion:\n"
            f"{example.question}"
        )

        print(
            f"\nOracle failure: "
            f"{diagnosis.failure_type.value}"
        )

        print(
            f"\nGold facts:"
        )

        for fact in sorted(
            gold
        ):
            print(
                f"  {fact}"
            )

        print(
            "\nMissing gold facts:"
        )

        for fact in sorted(
            missing
        ):
            print(
                f"  {fact}"
            )

        print(
            "\nInitial top-5 evidence:"
        )

        for result in results[:5]:

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

            gold_match = (
                get_retrieved_evidence(
                    [result]
                )
                & gold
            )

            print(
                f"\n  rank={result.rank}"
            )

            print(
                f"  title={title!r}"
            )

            print(
                f"  CE score="
                f"{result.score:.4f}"
            )

            print(
                f"  sentence_ids="
                f"{sentence_ids}"
            )

            print(
                f"  gold_match="
                f"{sorted(gold_match)}"
            )

            print(
                "  text="
                f"{result.chunk.text!r}"
            )

        print(
            "\nGenerated recovery queries:"
        )

        for query_index, query in enumerate(
            recovery_result.recovery_queries,
            start=1,
        ):
            print(
                f"  {query_index}. "
                f"{query}"
            )

        print()

        if inspected >= MAX_CASES:
            break

    # --------------------------------------------------------
    # Consistency
    # --------------------------------------------------------

    print(
        "=" * 100
    )

    print(
        "INSPECTION SUMMARY"
    )

    print("=" * 100)

    print(
        f"\nCoverage failures inspected: "
        f"{inspected}"
    )

    assert inspected == MAX_CASES

    print(
        "\nPASS: Inspected the requested "
        "number of real validation failures."
    )

    print(
        "PASS: Recovery queries were generated "
        "only from the original question and "
        "retrieved results."
    )

    print(
        "\nNOTE: Gold evidence is displayed only "
        "for offline inspection. It was not used "
        "to construct recovery queries."
    )


if __name__ == "__main__":
    main()