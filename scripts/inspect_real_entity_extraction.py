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
from src.recovery.entity_extraction import (
    ContentEntityExtractor,
    EntityExtractionConfig,
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

ENTITY_CONFIG = EntityExtractionConfig(
    model_name="dslim/bert-base-NER",
    source_result_depth=3,
    max_entities=15,
    aggregation_strategy="simple",
)


def main() -> None:

    print("=" * 100)
    print(
        "PHASE 10.3C-REAL — REAL CONTENT "
        "ENTITY EXTRACTION INSPECTION"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # Frozen validation data
    # --------------------------------------------------------

    print("\nLoading validation queries...")

    examples = load_query_examples(
        SPLIT_NAME
    )

    assert len(examples) == 200

    print("Loading frozen chunks...")

    chunks = load_frozen_chunks()

    assert len(chunks) == EXPECTED_CHUNKS

    print("Loading persisted embeddings...")

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    assert embeddings.shape == (
        EXPECTED_CHUNKS,
        EXPECTED_DIMENSION,
    )

    assert embeddings.dtype == np.float32

    retrieval_texts = [
        format_chunk_for_retrieval(chunk)
        for chunk in chunks
    ]

    # --------------------------------------------------------
    # Baseline retrieval
    # --------------------------------------------------------

    print("Constructing DenseRetriever...")

    dense_retriever = DenseRetriever(
        chunks=chunks,
        embeddings=embeddings,
    )

    print("Building BM25 index...")

    sparse_retriever = BM25Retriever(
        chunks=chunks,
        retrieval_texts=retrieval_texts,
    )

    print("Loading dense embedder...")

    embedder = DenseEmbedder(
        DenseEmbeddingConfig(
            model_name=DENSE_MODEL_NAME,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
        )
    )

    print("Loading cross-encoder...")

    reranker = CrossEncoderReranker(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        config=RerankerConfig(
            model_name=CROSS_ENCODER_MODEL_NAME,
            candidate_k=BASELINE_CANDIDATE_K,
            batch_size=BATCH_SIZE,
        ),
    )

    # --------------------------------------------------------
    # Real NER model
    # --------------------------------------------------------

    print(
        "Loading real NER model..."
    )

    extractor = ContentEntityExtractor(
        config=ENTITY_CONFIG
    )

    # --------------------------------------------------------
    # Inspect same type of real failures
    # --------------------------------------------------------

    inspected = 0

    print(
        "\nSearching for real evidence-coverage "
        "failures...\n"
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

        diagnosis = diagnose_oracle_retrieval(
            gold_evidence=gold,
            results=results,
        )

        if (
            diagnosis.failure_type
            != FailureType
            .EVIDENCE_COVERAGE_FAILURE
        ):
            continue

        entities = (
            extractor.extract_from_results(
                results
            )
        )

        missing = (
            gold
            - retrieved_evidence
        )

        inspected += 1

        print("=" * 100)
        print(f"CASE {inspected}")
        print("=" * 100)

        print(
            f"\nValidation position: "
            f"{position}"
        )

        print(
            f"\nQuestion:\n"
            f"{example.question}"
        )

        print(
            "\nMissing gold facts:"
        )

        for fact in sorted(missing):
            print(f"  {fact}")

        print(
            "\nTop-3 source passages:"
        )

        for result in results[:3]:

            title = (
                result.chunk.metadata.get(
                    "title",
                    "<unknown>",
                )
            )

            print(
                f"\n  rank={result.rank}"
            )

            print(
                f"  title={title!r}"
            )

            print(
                f"  text="
                f"{result.chunk.text!r}"
            )

        print(
            "\nExtracted candidate entities:"
        )

        if not entities:
            print("  <NONE>")

        for entity in entities:
            print(
                f"  text={entity.text!r} "
                f"label={entity.label!r} "
                f"score={entity.score:.4f} "
                f"source_rank="
                f"{entity.source_rank}"
            )

        print()

        if inspected >= MAX_CASES:
            break

    # --------------------------------------------------------
    # Checks
    # --------------------------------------------------------

    print("=" * 100)
    print("INSPECTION SUMMARY")
    print("=" * 100)

    print(
        f"\nCoverage failures inspected: "
        f"{inspected}"
    )

    assert inspected == MAX_CASES

    print(
        "\nPASS: Real NER executed over "
        "retrieved passage content."
    )

    print(
        "PASS: Entity extraction used no "
        "gold evidence, answers, or oracle "
        "labels."
    )

    print(
        "\nNOTE: Gold evidence is used only "
        "to select and display offline "
        "evaluation cases."
    )


if __name__ == "__main__":
    main()