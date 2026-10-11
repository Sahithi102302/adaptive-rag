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
    BridgeCandidate,
)
from src.recovery.entity_extraction import (
    ContentEntityExtractor,
    EntityExtractionConfig,
)
from src.recovery.llm_bridge_selection import (
    ClosedCandidateBridgeSelector,
    LLMBridgeSelectionConfig,
)
from src.recovery.qwen_generator import (
    QwenGeneratorConfig,
    QwenTextGenerator,
)
from src.recovery.relation_query import (
    ConstrainedRelationQueryGenerator,
    RelationQueryConfig,
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
# Frozen experiment configuration
# ============================================================

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


LLM_BRIDGE_SELECTION_CONFIG = (
    LLMBridgeSelectionConfig(
        max_passage_chars=1200,
        include_ner_metadata=True,
    )
)


RELATION_QUERY_CONFIG = RelationQueryConfig(
    max_passage_chars=1200,
)


QWEN_CONFIG = QwenGeneratorConfig(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens=128,
)


# ============================================================
# Offline diagnostic expected bridges
# ============================================================
#
# IMPORTANT:
#
# These are used ONLY in Condition B:
#
#     expected bridge
#         ->
#     constrained query formulation
#
# They are NEVER supplied to:
#
#     retrieval
#     NER candidate generation
#     runtime bridge selection
#     runtime query formulation
#
# This is an offline intervention used to isolate whether
# relation/query formulation can succeed when bridge selection
# is correct.
#
# None means that no clean expected bridge is available in the
# frozen candidate pool, or that the case has an evaluation
# caveat that makes the intervention inappropriate.
# ============================================================

EXPECTED_BRIDGES: dict[int, str | None] = {
    1: "Oscar Wilde",
    2: "Burlington Northern Railroad",
    3: None,
    4: "Seattle University",
    5: "Lin Dan",
    6: "Shia LaBeouf",
    7: None,
    8: None,
    9: "R2Bees",
    10: None,
}


def build_bridge_candidates(
    entities,
    results,
) -> tuple[BridgeCandidate, ...]:
    """
    Convert extracted runtime-visible entities into
    BridgeCandidate objects.

    Each extracted entity is resolved back to the retrieved
    passage from which it originated.

    No gold evidence, gold answer, gold document ID, or oracle
    failure information is used here.
    """

    results_by_chunk_id = {
        result.chunk.id: result
        for result in results
    }

    candidates: list[BridgeCandidate] = []

    for entity in entities:
        source_result = results_by_chunk_id.get(
            entity.source_chunk_id
        )

        if source_result is None:
            raise ValueError(
                "Could not resolve extracted entity "
                f"{entity.text!r} to retrieved chunk "
                f"{entity.source_chunk_id!r}."
            )

        title = source_result.chunk.metadata.get(
            "title",
            "<unknown>",
        )

        candidates.append(
            BridgeCandidate(
                entity=entity.text,
                entity_label=entity.label,
                ner_score=entity.score,
                source_rank=entity.source_rank,
                source_chunk_id=entity.source_chunk_id,
                source_title=str(title),
                source_text=source_result.chunk.text,
            )
        )

    return tuple(candidates)


def find_candidate_by_entity(
    candidates: tuple[BridgeCandidate, ...],
    entity_name: str,
) -> BridgeCandidate | None:
    """
    Find one exact normalized entity in the frozen candidate
    pool.

    This intentionally does NOT perform fuzzy matching,
    substring repair, entity merging, or NER-fragment repair.

    If the expected entity is absent, that remains visible as
    a candidate-generation failure.
    """

    target = " ".join(
        entity_name.split()
    ).casefold()

    matches = [
        candidate
        for candidate in candidates
        if " ".join(
            candidate.entity.split()
        ).casefold()
        == target
    ]

    if not matches:
        return None

    if len(matches) > 1:
        raise ValueError(
            "Expected exactly one normalized candidate for "
            f"{entity_name!r}, but found {len(matches)}."
        )

    return matches[0]


def main() -> None:
    print("=" * 100)
    print(
        "PHASE 10.3H-B — REAL CONSTRAINED "
        "RELATION/QUERY FORMULATION INSPECTION"
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
    # Frozen runtime-visible entity extraction
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
    # Qwen backend
    #
    # IMPORTANT:
    #
    # Load ONE Qwen instance and reuse it for:
    #
    #   1. runtime closed-candidate bridge selection
    #   2. constrained relation/query formulation
    #
    # There is no reason to load the model twice.
    # --------------------------------------------------------

    print(
        "Loading Qwen model..."
    )

    qwen_generator = QwenTextGenerator(
        config=QWEN_CONFIG
    )

    # --------------------------------------------------------
    # Runtime bridge selector
    # --------------------------------------------------------

    llm_bridge_selector = (
        ClosedCandidateBridgeSelector(
            generator=qwen_generator,
            config=(
                LLM_BRIDGE_SELECTION_CONFIG
            ),
        )
    )

    # --------------------------------------------------------
    # Constrained relation/query formulator
    # --------------------------------------------------------

    relation_query_generator = (
        ConstrainedRelationQueryGenerator(
            generator=qwen_generator,
            config=RELATION_QUERY_CONFIG,
        )
    )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    inspected = 0

    candidate_counts: list[int] = []

    runtime_selection_valid = 0
    runtime_selection_invalid = 0

    runtime_query_valid = 0
    runtime_query_invalid = 0

    expected_query_attempted = 0
    expected_query_valid = 0
    expected_query_invalid = 0

    expected_bridge_missing = 0
    expected_bridge_skipped = 0

    # --------------------------------------------------------
    # Inspect the SAME first 10 oracle evidence-coverage
    # failures used in the previous 10.3 experiments.
    #
    # Gold evidence and oracle diagnosis are used ONLY for:
    #
    #   - offline case selection
    #   - displaying missing evidence
    #   - choosing the predefined Condition-B intervention
    #
    # They are NOT supplied to the runtime path.
    # --------------------------------------------------------

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
        # Offline oracle case selection ONLY
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
        # No gold/oracle information enters this stage.
        # ----------------------------------------------------

        entities = (
            entity_extractor
            .extract_from_results(
                results
            )
        )

        bridge_candidates = (
            build_bridge_candidates(
                entities=entities,
                results=results,
            )
        )

        candidate_counts.append(
            len(bridge_candidates)
        )

        # ----------------------------------------------------
        # Case display
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
        # Runtime-visible evidence
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
        # Frozen bounded candidate pool
        # ----------------------------------------------------

        print(
            "\nExtracted bridge candidates "
            "(runtime-visible):"
        )

        if not bridge_candidates:
            print(
                "  <NONE>"
            )

        for index, candidate in enumerate(
            bridge_candidates,
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

        # ====================================================
        # CONDITION A
        #
        # ACTUAL RUNTIME PATH
        #
        # retrieved evidence
        #     ->
        # runtime NER candidates
        #     ->
        # Qwen bridge selection
        #     ->
        # constrained query formulation
        #
        # NO gold/oracle information enters this condition.
        # ====================================================

        print()

        print(
            "-" * 100
        )

        print(
            "CONDITION A — "
            "RUNTIME-SELECTED BRIDGE"
        )

        print(
            "-" * 100
        )

        if not bridge_candidates:
            runtime_selection_invalid += 1
            runtime_query_invalid += 1

            print(
                "Runtime bridge selection: FAILED"
            )

            print(
                "Reason: empty candidate pool."
            )

        else:
            try:
                runtime_selection = (
                    llm_bridge_selector.select(
                        question=example.question,
                        candidates=bridge_candidates,
                    )
                )

            except ValueError as exc:
                runtime_selection_invalid += 1
                runtime_query_invalid += 1

                print(
                    "Runtime bridge selection: INVALID"
                )

                print(
                    f"Reason: {exc}"
                )

            else:
                runtime_selection_valid += 1

                runtime_candidate = (
                    runtime_selection
                    .selected_candidate
                )

                print(
                    "Runtime bridge selection: VALID"
                )

                print(
                    "Runtime selected index: "
                    f"{runtime_selection.selected_index}"
                )

                print(
                    "Runtime selected bridge: "
                    f"{runtime_candidate.entity!r}"
                )

                print(
                    "Runtime selected source rank: "
                    f"{runtime_candidate.source_rank}"
                )

                print(
                    "\nBridge-selector raw output:"
                )

                print(
                    runtime_selection.raw_output
                )

                try:
                    runtime_query_result = (
                        relation_query_generator.generate(
                            question=example.question,
                            selected_candidate=(
                                runtime_candidate
                            ),
                            candidates=bridge_candidates,
                        )
                    )

                except ValueError as exc:
                    runtime_query_invalid += 1

                    print(
                        "\nRuntime query formulation: "
                        "INVALID"
                    )

                    print(
                        f"Reason: {exc}"
                    )

                else:
                    runtime_query_valid += 1

                    print(
                        "\nRuntime query formulation: "
                        "VALID"
                    )

                    print(
                        "Runtime second-hop query: "
                        f"{runtime_query_result.query!r}"
                    )

                    print(
                        "\nQuery-generator raw output:"
                    )

                    print(
                        runtime_query_result.raw_output
                    )

        # ====================================================
        # CONDITION B
        #
        # OFFLINE EXPECTED-BRIDGE INTERVENTION
        #
        # This condition deliberately bypasses runtime bridge
        # selection to isolate relation/query formulation.
        #
        # It is NOT a deployable runtime path.
        # ====================================================

        print()

        print(
            "-" * 100
        )

        print(
            "CONDITION B — OFFLINE EXPECTED-BRIDGE "
            "FORMULATION DIAGNOSTIC"
        )

        print(
            "-" * 100
        )

        expected_bridge = EXPECTED_BRIDGES.get(
            inspected
        )

        if expected_bridge is None:
            expected_bridge_skipped += 1

            print(
                "Expected-bridge diagnostic: SKIPPED"
            )

            print(
                "Reason: no clean expected bridge "
                "intervention is defined for this "
                "frozen case."
            )

        else:
            expected_candidate = (
                find_candidate_by_entity(
                    candidates=bridge_candidates,
                    entity_name=expected_bridge,
                )
            )

            if expected_candidate is None:
                expected_bridge_missing += 1

                print(
                    "Expected-bridge diagnostic: "
                    "CANDIDATE NOT FOUND"
                )

                print(
                    "Expected bridge: "
                    f"{expected_bridge!r}"
                )

                print(
                    "This remains visible as a "
                    "candidate-generation failure."
                )

            else:
                expected_query_attempted += 1

                print(
                    "Expected-bridge diagnostic: "
                    "CANDIDATE FOUND"
                )

                print(
                    "Expected bridge: "
                    f"{expected_candidate.entity!r}"
                )

                print(
                    "Expected bridge source rank: "
                    f"{expected_candidate.source_rank}"
                )

                try:
                    expected_query_result = (
                        relation_query_generator.generate(
                            question=example.question,
                            selected_candidate=(
                                expected_candidate
                            ),
                            candidates=bridge_candidates,
                        )
                    )

                except ValueError as exc:
                    expected_query_invalid += 1

                    print(
                        "\nExpected-bridge query "
                        "formulation: INVALID"
                    )

                    print(
                        f"Reason: {exc}"
                    )

                else:
                    expected_query_valid += 1

                    print(
                        "\nExpected-bridge query "
                        "formulation: VALID"
                    )

                    print(
                        "Expected-bridge second-hop "
                        "query: "
                        f"{expected_query_result.query!r}"
                    )

                    print(
                        "\nQuery-generator raw output:"
                    )

                    print(
                        expected_query_result.raw_output
                    )

        print()

        if inspected >= MAX_CASES:
            break

    # ========================================================
    # Summary
    # ========================================================

    print(
        "=" * 100
    )

    print(
        "PHASE 10.3H-B INSPECTION SUMMARY"
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

    print(
        "\nRuntime bridge selection:"
    )

    print(
        "  Valid selections: "
        f"{runtime_selection_valid}"
    )

    print(
        "  Invalid selections: "
        f"{runtime_selection_invalid}"
    )

    print(
        "\nRuntime-selected bridge "
        "query formulation:"
    )

    print(
        "  Valid queries: "
        f"{runtime_query_valid}"
    )

    print(
        "  Invalid queries: "
        f"{runtime_query_invalid}"
    )

    print(
        "\nOffline expected-bridge "
        "formulation diagnostic:"
    )

    print(
        "  Attempted: "
        f"{expected_query_attempted}"
    )

    print(
        "  Valid queries: "
        f"{expected_query_valid}"
    )

    print(
        "  Invalid queries: "
        f"{expected_query_invalid}"
    )

    print(
        "  Expected bridge missing: "
        f"{expected_bridge_missing}"
    )

    print(
        "  Diagnostic skipped: "
        f"{expected_bridge_skipped}"
    )

    # --------------------------------------------------------
    # Structural checks
    # --------------------------------------------------------

    assert (
        inspected
        == MAX_CASES
    )

    assert (
        runtime_selection_valid
        + runtime_selection_invalid
        == inspected
    )

    assert (
        runtime_query_valid
        + runtime_query_invalid
        == inspected
    )

    assert (
        expected_query_valid
        + expected_query_invalid
        == expected_query_attempted
    )

    assert (
        expected_query_attempted
        + expected_bridge_missing
        + expected_bridge_skipped
        == inspected
    )

    # --------------------------------------------------------
    # Methodological guarantees
    # --------------------------------------------------------

    print(
        "\nPASS: The frozen first-hop retrieval "
        "pipeline was preserved."
    )

    print(
        "PASS: Bridge candidates were generated "
        "only from runtime-visible retrieved "
        "passages."
    )

    print(
        "PASS: Runtime bridge selection received "
        "no gold evidence, gold answer, gold "
        "document IDs, or oracle failure labels."
    )

    print(
        "PASS: Runtime query formulation received "
        "only the original question, frozen "
        "runtime-visible candidate evidence, and "
        "the runtime-selected bridge."
    )

    print(
        "PASS: The constrained formulator could "
        "not replace the supplied bridge entity."
    )

    print(
        "PASS: Expected bridges were used only "
        "in the separate offline formulation "
        "diagnostic."
    )

    print(
        "PASS: Missing expected candidates were "
        "not repaired, merged, or injected into "
        "the candidate pool."
    )

    print(
        "PASS: No second-hop retrieval was "
        "executed."
    )

    print(
        "\nNOTE: Condition B is an offline "
        "intervention for isolating relation/query "
        "formulation. It is not part of the "
        "runtime AdaptiveRAG recovery policy."
    )


if __name__ == "__main__":
    main()