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


LLM_BRIDGE_SELECTION_CONFIG = (
    LLMBridgeSelectionConfig(
        max_passage_chars=1200,
        include_ner_metadata=True,
    )
)


QWEN_CONFIG = QwenGeneratorConfig(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens=128,
)


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


def main() -> None:
    print("=" * 100)
    print(
        "PHASE 10.3G-E — REAL CLOSED-CANDIDATE "
        "LLM BRIDGE-SELECTION INSPECTION"
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
    # Runtime-visible closed-candidate LLM selection
    #
    # IMPORTANT:
    # Qwen is loaded once and reused across all inspected cases.
    #
    # The model receives:
    #   - original question
    #   - retrieved passages
    #   - bounded candidate entities
    #
    # It does NOT receive:
    #   - gold evidence
    #   - gold answer
    #   - gold document IDs
    #   - oracle failure type
    # --------------------------------------------------------

    print(
        "Loading Qwen bridge-selection model..."
    )

    qwen_generator = QwenTextGenerator(
        config=QWEN_CONFIG
    )

    llm_bridge_selector = (
        ClosedCandidateBridgeSelector(
            generator=qwen_generator,
            config=(
                LLM_BRIDGE_SELECTION_CONFIG
            ),
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
    # They are never passed into entity extraction or LLM
    # bridge selection.
    # --------------------------------------------------------

    inspected = 0

    candidate_counts: list[int] = []

    valid_selections = 0
    invalid_selections = 0

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
        # Display case information before selection.
        #
        # This ensures candidate-generation failures remain
        # visible even if Qwen returns malformed output.
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
        # Full bounded candidate pool
        #
        # Qwen receives this entire pool.
        #
        # The MS MARCO bridge-selection cross-encoder is NOT
        # used to prune candidates in this experiment.
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

        # ----------------------------------------------------
        # Runtime-visible closed-candidate LLM selection
        #
        # Inputs:
        #   original question
        #   retrieved source passages
        #   supplied bridge candidates
        #
        # Output:
        #   exactly one supplied candidate index
        #
        # No gold/oracle information is supplied.
        # ----------------------------------------------------

        if not bridge_candidates:
            invalid_selections += 1

            print(
                "\nClosed-candidate LLM selection:"
            )

            print(
                "  FAILED: empty candidate pool."
            )

            print()

            if inspected >= MAX_CASES:
                break

            continue

        try:
            selection = (
                llm_bridge_selector.select(
                    question=example.question,
                    candidates=bridge_candidates,
                )
            )

        except ValueError as exc:
            invalid_selections += 1

            print(
                "\nClosed-candidate LLM selection:"
            )

            print(
                f"  FAILED: {exc}"
            )

            print()

            if inspected >= MAX_CASES:
                break

            continue

        valid_selections += 1

        print(
            "\nClosed-candidate LLM selection:"
        )

        print(
            f"  selected_index="
            f"{selection.selected_index}"
        )

        print(
            f"  selected_entity="
            f"{selection.selected_candidate.entity!r}"
        )

        print(
            f"  entity_label="
            f"{selection.selected_candidate.entity_label!r}"
        )

        print(
            f"  source_rank="
            f"{selection.selected_candidate.source_rank}"
        )

        print(
            f"  source_chunk_id="
            f"{selection.selected_candidate.source_chunk_id!r}"
        )

        print(
            "\nRaw model output:"
        )

        print(
            selection.raw_output
        )

        print()

        if inspected >= MAX_CASES:
            break

    # --------------------------------------------------------
    # Checks / summary
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

    print(
        "Valid closed-candidate selections: "
        f"{valid_selections}"
    )

    print(
        "Invalid/unparseable selections: "
        f"{invalid_selections}"
    )

    assert (
        inspected
        == MAX_CASES
    )

    assert (
        valid_selections
        + invalid_selections
        == inspected
    )

    print(
        "\nPASS: Bridge candidates were "
        "extracted only from runtime-visible "
        "retrieved passages."
    )

    print(
        "PASS: Closed-candidate LLM selection "
        "received only the original question, "
        "runtime-visible retrieved passages, "
        "and supplied candidate entities."
    )

    print(
        "PASS: The LLM could select only an "
        "entity from the supplied bounded "
        "candidate pool."
    )

    print(
        "PASS: Gold evidence, gold answers, "
        "gold document IDs, and oracle "
        "failure labels were not supplied "
        "to LLM bridge selection."
    )

    print(
        "PASS: The MS MARCO bridge-selection "
        "cross-encoder was not used to prune "
        "the candidate pool."
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