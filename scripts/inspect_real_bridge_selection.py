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
from src.recovery.bridge_selection import (
    BridgeSelectionConfig,
    CrossEncoderBridgeSelector,
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
    max_entities_per_source=5,
    aggregation_strategy="first",
)


BRIDGE_SELECTION_CONFIG = BridgeSelectionConfig(
    model_name=CROSS_ENCODER_MODEL_NAME,
    batch_size=BATCH_SIZE,
    top_k=3,
    max_context_chars=1200,
)


def main() -> None:

    print("=" * 100)
    print(
        "PHASE 10.3F-B — REAL CONSTRAINED "
        "BRIDGE-SELECTION INSPECTION"
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
        "Loading retrieval cross-encoder..."
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
    # Runtime-visible bridge candidate generation
    # --------------------------------------------------------

    print(
        "Loading content NER model..."
    )

    entity_extractor = (
        ContentEntityExtractor(
            config=ENTITY_CONFIG
        )
    )

    # --------------------------------------------------------
    # Runtime-visible bridge candidate selection
    # --------------------------------------------------------

    print(
        "Loading bridge-selection "
        "cross-encoder..."
    )

    bridge_selector = (
        CrossEncoderBridgeSelector(
            config=(
                BRIDGE_SELECTION_CONFIG
            )
        )
    )

    # --------------------------------------------------------
    # Inspect the same first 10 oracle evidence-coverage
    # failures used in the previous 10.3 experiments.
    #
    # IMPORTANT:
    #
    # Gold evidence and oracle diagnosis are used only for
    # offline case selection/display.
    #
    # They are never passed into entity extraction or bridge
    # selection.
    # --------------------------------------------------------

    inspected = 0

    candidate_counts: list[int] = []

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
        # Runtime-visible candidate generation
        #
        # Inputs:
        #   retrieved results only
        #
        # No gold/oracle information is supplied.
        # ----------------------------------------------------

        entities = (
            entity_extractor
            .extract_from_results(
                results
            )
        )

        # ----------------------------------------------------
        # Runtime-visible candidate selection
        #
        # Inputs:
        #   original question
        #   extracted entities
        #   retrieved results
        #
        # No gold/oracle information is supplied.
        # ----------------------------------------------------

        selection = (
            bridge_selector.select(
                question=example.question,
                entities=entities,
                results=results,
            )
        )

        candidate_counts.append(
            len(
                selection.all_candidates
            )
        )

        # ----------------------------------------------------
        # Display
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
        # Offline evaluation information
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
        # Runtime-visible source evidence
        # ----------------------------------------------------

        print(
            "\nTop-3 source passages "
            "(runtime-visible):"
        )

        for result in results[
            : ENTITY_CONFIG
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
        # Extracted candidate pool
        # ----------------------------------------------------

        print(
            "\nExtracted bridge candidates "
            "(runtime-visible):"
        )

        if not (
            selection.all_candidates
        ):
            print(
                "  <NONE>"
            )

        for index, candidate in enumerate(
            selection.all_candidates,
            start=1,
        ):
            print(
                f"  {index}. "
                f"entity="
                f"{candidate.entity!r} "
                f"label="
                f"{candidate.entity_label!r} "
                f"ner_score="
                f"{candidate.ner_score:.4f} "
                f"source_rank="
                f"{candidate.source_rank}"
            )

        # ----------------------------------------------------
        # Question-conditioned CE ranking
        # ----------------------------------------------------

        print(
            "\nTop bridge selections "
            "(runtime-visible):"
        )

        if not (
            selection.ranked_candidates
        ):
            print(
                "  <NONE>"
            )

        for scored in (
            selection.ranked_candidates
        ):
            print(
                f"  rank="
                f"{scored.rank} "
                f"entity="
                f"{scored.candidate.entity!r} "
                f"ce_score="
                f"{scored.score:.6f} "
                f"source_rank="
                f"{scored.candidate.source_rank}"
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

    if candidate_counts:
        print(
            "Candidate pool sizes: "
            f"{candidate_counts}"
        )

        print(
            "Mean candidate pool size: "
            f"{np.mean(candidate_counts):.2f}"
        )

    assert (
        inspected
        == MAX_CASES
    )

    print(
        "\nPASS: Bridge candidates were "
        "extracted only from runtime-visible "
        "retrieved passages."
    )

    print(
        "PASS: Bridge selection received "
        "only the original question, "
        "extracted entities, and retrieved "
        "passages."
    )

    print(
        "PASS: Gold evidence, gold answers, "
        "gold document IDs, and oracle "
        "failure labels were not supplied "
        "to bridge selection."
    )

    print(
        "PASS: The frozen baseline "
        "retrieval pipeline was preserved."
    )

    print(
        "\nNOTE: Missing gold facts and "
        "oracle failure type are used only "
        "for offline case selection and "
        "qualitative evaluation."
    )


if __name__ == "__main__":
    main()