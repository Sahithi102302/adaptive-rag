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
from src.recovery.bridge_query import (
    BridgeQueryConfig,
    RelationAwareBridgeQueryGenerator,
)
from src.recovery.qwen_generator import (
    QwenGeneratorConfig,
    QwenTextGenerator,
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


SPLIT_NAME = "validation"

BASELINE_CANDIDATE_K = 20
FINAL_K = 10
BATCH_SIZE = 32

MAX_CASES = 10


BRIDGE_CONFIG = BridgeQueryConfig(
    source_result_depth=3,
    max_queries=3,
    max_passage_chars=1200,
)


QWEN_CONFIG = QwenGeneratorConfig(
    model_name=(
        "Qwen/Qwen2.5-1.5B-Instruct"
    ),
    max_new_tokens=128,
)


def main() -> None:

    print("=" * 100)
    print(
        "PHASE 10.3D-REAL — RELATION-AWARE "
        "BRIDGE QUERY INSPECTION"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # Frozen validation data
    # --------------------------------------------------------

    print(
        "\nLoading validation queries..."
    )

    examples = load_query_examples(
        SPLIT_NAME
    )

    assert len(examples) == 200

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

    assert (
        embeddings.dtype
        == np.float32
    )

    retrieval_texts = [
        format_chunk_for_retrieval(
            chunk
        )
        for chunk in chunks
    ]

    # --------------------------------------------------------
    # Frozen baseline retrieval
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

    # --------------------------------------------------------
    # Relation-aware bridge-query model
    # --------------------------------------------------------

    print(
        "Loading Qwen bridge-query model..."
    )

    text_generator = QwenTextGenerator(
        config=QWEN_CONFIG
    )

    bridge_generator = (
        RelationAwareBridgeQueryGenerator(
            text_generator=(
                text_generator
            ),
            config=BRIDGE_CONFIG,
        )
    )

    # --------------------------------------------------------
    # Inspect the same type of real failures used
    # in the Phase 10.3C NER experiment.
    #
    # IMPORTANT:
    #
    # Gold evidence and the oracle diagnosis are used
    # only to select/evaluate offline cases.
    #
    # They are NOT passed into the bridge-query
    # generator.
    # --------------------------------------------------------

    inspected = 0

    print(
        "\nSearching for real "
        "evidence-coverage failures...\n"
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

        # ----------------------------------------------------
        # Frozen first-hop retrieval
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Offline oracle case selection only
        # ----------------------------------------------------

        diagnosis = (
            diagnose_oracle_retrieval(
                gold_evidence=gold,
                results=results,
            )
        )

        if (
            diagnosis.failure_type
            != FailureType
            .EVIDENCE_COVERAGE_FAILURE
        ):
            continue

        missing = (
            gold
            - retrieved_evidence
        )

        inspected += 1

        # ----------------------------------------------------
        # Display case BEFORE generation.
        #
        # This makes it visually clear what information
        # exists in the runtime-visible first-hop evidence.
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Offline evaluation information.
        #
        # This is deliberately printed separately from
        # generation and is never included in the prompt.
        # ----------------------------------------------------

        print(
            "\nMissing gold facts "
            "(offline evaluation only):"
        )

        for fact in sorted(
            missing
        ):
            print(
                f"  {fact}"
            )

        # ----------------------------------------------------
        # Runtime-visible evidence
        # ----------------------------------------------------

        print(
            "\nTop-3 source passages "
            "(runtime-visible):"
        )

        for result in results[
            : BRIDGE_CONFIG
            .source_result_depth
        ]:

            title = (
                result.chunk.metadata.get(
                    "title",
                    "<unknown>",
                )
            )

            print(
                f"\n  rank="
                f"{result.rank}"
            )

            print(
                f"  title="
                f"{title!r}"
            )

            print(
                f"  text="
                f"{result.chunk.text!r}"
            )

        # ----------------------------------------------------
        # Relation-aware generation
        #
        # ONLY:
        #   original question
        #   retrieved results
        #
        # are passed to the generator.
        # ----------------------------------------------------

        bridge_result = (
            bridge_generator.generate(
                question=(
                    example.question
                ),
                results=results,
            )
        )

        print(
            "\nParsed bridge queries:"
        )

        if not bridge_result.queries:
            print(
                "  <NONE>"
            )

        for query_index, query in enumerate(
            bridge_result.queries,
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
    # Checks
    # --------------------------------------------------------

    print(
        "=" * 100
    )

    print(
        "INSPECTION SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"\nCoverage failures inspected: "
        f"{inspected}"
    )

    assert (
        inspected
        == MAX_CASES
    )

    print(
        "\nPASS: Relation-aware bridge "
        "queries were generated from the "
        "original question and retrieved "
        "passages."
    )

    print(
        "PASS: Bridge-query generation "
        "received no gold evidence, gold "
        "answers, gold document IDs, or "
        "oracle failure labels."
    )

    print(
        "PASS: The frozen baseline "
        "retrieval pipeline was preserved."
    )

    print(
        "\nNOTE: Gold evidence and oracle "
        "failure type are used only to "
        "select and display offline "
        "evaluation cases."
    )


if __name__ == "__main__":
    main()