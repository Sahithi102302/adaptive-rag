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
from src.recovery.second_hop import (
    SecondHopRetrievalConfig,
    SecondHopRetriever,
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
from src.recovery.evidence_selection import (
    EvidencePreservingSelector,
    EvidenceSelectionConfig,
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
SECOND_HOP_CONFIG = SecondHopRetrievalConfig(
    top_k=FINAL_K,
)
SELECTION_ALLOCATIONS = {
    "Hop-balanced 5/5": (5, 5),
    "Hop-balanced 7/3": (7, 3),
    "Hop-balanced 3/7": (3, 7),
}
# ============================================================
# Offline diagnostic expected bridges
# ============================================================
#
# IMPORTANT:
#
# These values are used ONLY in Condition B.
#
# They are never supplied to:
#   - first-hop retrieval,
#   - runtime NER,
#   - runtime bridge selection,
#   - runtime query formulation,
#   - second-hop retrieval itself.
#
# Gold evidence is used only AFTER retrieval to score whether
# the previously missing evidence was recovered.
#
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
# ============================================================
# Helpers
# ============================================================
def build_bridge_candidates(
    entities,
    results,
) -> tuple[BridgeCandidate, ...]:
    """
    Convert runtime-visible extracted entities into bridge
    candidates.
    No gold/oracle information is used.
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
                source_chunk_id=(
                    entity.source_chunk_id
                ),
                source_title=str(title),
                source_text=(
                    source_result.chunk.text
                ),
            )
        )
    return tuple(candidates)
def find_candidate_by_entity(
    candidates: tuple[BridgeCandidate, ...],
    entity_name: str,
) -> BridgeCandidate | None:
    """
    Exact normalized lookup only.
    No fuzzy repair, substring repair, entity merging,
    or NER-fragment repair is allowed.
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
def get_recovered_missing_facts(
    missing,
    second_hop_results,
):
    """
    Offline evaluation ONLY.
    Determine which facts that were absent from the original
    first-hop final top-k appear in the second-hop final top-k.
    """
    second_hop_evidence = (
        get_retrieved_evidence(
            second_hop_results
        )
    )
    return (
        missing
        & second_hop_evidence
    )
def get_recovered_fact_ranks(
    recovered_facts,
    second_hop_results,
) -> dict[tuple[str, int], list[int]]:
    """
    Offline evaluation ONLY.
    Record the second-hop final ranks at which each recovered
    supporting fact appears.
    A fact may map to more than one returned chunk because
    sentence-aware chunks can overlap.
    """
    fact_ranks: dict[
        tuple[str, int],
        list[int],
    ] = {
        fact: []
        for fact in recovered_facts
    }
    for result in second_hop_results:
        evidence = get_retrieved_evidence(
            [result]
        )
        for fact in recovered_facts:
            if fact in evidence:
                fact_ranks[fact].append(
                    result.rank
                )
    return fact_ranks
def print_second_hop_results(
    results,
    limit: int = 5,
) -> None:
    """
    Display a small runtime-visible sample of second-hop
    retrieval results.
    """
    if not results:
        print("  <NONE>")
        return
    for result in results[:limit]:
        title = result.chunk.metadata.get(
            "title",
            "<unknown>",
        )
        print(
            f"\n  rank={result.rank}"
        )
        print(
            f"  score={result.score:.4f}"
        )
        print(
            f"  title={title!r}"
        )
        print(
            f"  text={result.chunk.text!r}"
        )
def print_recovery_evaluation(
    missing,
    second_hop_results,
) -> tuple[int, bool]:
    """
    Print offline recovery metrics for one second-hop attempt.
    Returns:
        recovered fact count,
        whether all originally missing facts were recovered.
    """
    recovered = (
        get_recovered_missing_facts(
            missing=missing,
            second_hop_results=(
                second_hop_results
            ),
        )
    )
    ranks = get_recovered_fact_ranks(
        recovered_facts=recovered,
        second_hop_results=(
            second_hop_results
        ),
    )
    complete = (
        bool(missing)
        and missing.issubset(recovered)
    )
    print(
        "\nOffline recovery evaluation:"
    )
    print(
        "  Originally missing facts: "
        f"{len(missing)}"
    )
    print(
        "  Recovered missing facts: "
        f"{len(recovered)}"
    )
    print(
        "  Any missing fact recovered: "
        f"{bool(recovered)}"
    )
    print(
        "  All missing facts recovered: "
        f"{complete}"
    )
    if recovered:
        print(
            "  Recovered fact ranks:"
        )
        for fact in sorted(recovered):
            print(
                f"    {fact}: "
                f"{ranks[fact]}"
            )
    else:
        print(
            "  Recovered fact ranks: <NONE>"
        )
    return (
        len(recovered),
        complete,
    )
def select_policy_results(
    policy_name,
    first_hop_results,
    second_hop_results,
):
    """
    Select final evidence without using gold annotations.
    Return original retrieval result objects so the existing
    evidence evaluator can inspect their sentence metadata.
    """
    if policy_name in (
        "First-hop only",
        "Naive rank union",
    ):
        if policy_name == "First-hop only":
            candidates = first_hop_results
        else:
            candidates = (
                *first_hop_results,
                *second_hop_results,
            )
        selected = []
        seen = set()
        for result in candidates:
            chunk_id = result.chunk.id
            if chunk_id in seen:
                continue
            if len(selected) >= FINAL_K:
                break
            seen.add(chunk_id)
            selected.append(result)
        return tuple(selected)
    first_slots, second_slots = (
        SELECTION_ALLOCATIONS[policy_name]
    )
    selector = EvidencePreservingSelector(
        EvidenceSelectionConfig(
            final_k=FINAL_K,
            reserved_first_hop=first_slots,
            reserved_second_hop=second_slots,
        )
    )
    selection = selector.select(
        original_results=first_hop_results,
        recovered_results=second_hop_results,
    )
    by_id = {}
    for result in (
        *first_hop_results,
        *second_hop_results,
    ):
        by_id.setdefault(
            result.chunk.id,
            result,
        )
    return tuple(
        by_id[item.chunk_id]
        for item in selection.selected_results
    )
POLICY_NAMES = (
    "First-hop only",
    "Naive rank union",
    "Hop-balanced 5/5",
    "Hop-balanced 7/3",
    "Hop-balanced 3/7",
)
def make_policy_counters():
    return {
        name: {
            "evaluated": 0,
            "gold_total": 0,
            "gold_retrieved": 0,
            "complete": 0,
            "recovered": 0,
            "lost": 0,
            "repaired_cases": 0,
            "harmed_cases": 0,
        }
        for name in POLICY_NAMES
    }
def evaluate_selection_policies(
    *,
    condition,
    gold,
    first_hop_results,
    second_hop_results,
    counters,
):
    """
    Gold annotations are used only after selection.
    Recovered facts are previously missing facts that appear
    in the final selected context.
    Lost facts are originally retrieved gold facts absent
    from the final selected context.
    """
    first_evidence = (
        get_retrieved_evidence(first_hop_results)
        & gold
    )
    missing = gold - first_evidence
    print(
        f"\n{condition}: final-context policy comparison"
    )
    for policy_name in POLICY_NAMES:
        selected = select_policy_results(
            policy_name,
            first_hop_results,
            second_hop_results,
        )
        selected_ids = [
            result.chunk.id
            for result in selected
        ]
        assert len(selected) <= FINAL_K
        assert len(selected_ids) == len(set(selected_ids))
        final_evidence = (
            get_retrieved_evidence(selected)
            & gold
        )
        recovered = missing & final_evidence
        lost = first_evidence - final_evidence
        complete = gold.issubset(final_evidence)
        stats = counters[policy_name]
        stats["evaluated"] += 1
        stats["gold_total"] += len(gold)
        stats["gold_retrieved"] += len(final_evidence)
        stats["complete"] += int(complete)
        stats["recovered"] += len(recovered)
        stats["lost"] += len(lost)
        if complete and not gold.issubset(first_evidence):
            stats["repaired_cases"] += 1
        if lost:
            stats["harmed_cases"] += 1
        print(
            f"  {policy_name}: "
            f"coverage={len(final_evidence)}/{len(gold)}, "
            f"recovered={len(recovered)}, "
            f"lost={len(lost)}, "
            f"complete={complete}"
        )
def print_policy_summary(condition, counters):
    print("\n" + "=" * 90)
    print(f"{condition} — FINAL CONTEXT SELECTION SUMMARY")
    print("=" * 90)
    for policy_name, stats in counters.items():
        denominator = stats["gold_total"]
        recall = (
            stats["gold_retrieved"] / denominator
            if denominator
            else 0.0
        )
        print(f"\n{policy_name}")
        print(f"  Evaluated cases: {stats['evaluated']}")
        print(f"  Supporting-fact recall: {recall:.4f}")
        print(f"  Complete cases: {stats['complete']}")
        print(f"  Newly recovered facts: {stats['recovered']}")
        print(f"  Original facts lost: {stats['lost']}")
        print(f"  Repaired cases: {stats['repaired_cases']}")
        print(f"  Cases losing original facts: {stats['harmed_cases']}")
# ============================================================
# Main
# ============================================================
def main() -> None:
    print("=" * 100)
    print(
        "PHASE 10.4C — REAL CONTROLLED "
        "SECOND-HOP RETRIEVAL"
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
    assert embeddings.dtype == np.float32
    retrieval_texts = [
        format_chunk_for_retrieval(
            chunk
        )
        for chunk in chunks
    ]
    # --------------------------------------------------------
    # Frozen retrieval stack
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
    # Second-hop retrieval wrapper
    # --------------------------------------------------------
    second_hop_retriever = (
        SecondHopRetriever(
            reranker=reranker,
            embedder=embedder,
            config=SECOND_HOP_CONFIG,
        )
    )
    # --------------------------------------------------------
    # Frozen NER
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
    # One shared Qwen backend
    # --------------------------------------------------------
    print(
        "Loading Qwen model..."
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
    total_missing_facts = 0
    # Runtime path
    runtime_query_valid = 0
    runtime_query_invalid = 0
    runtime_retrieval_attempted = 0
    runtime_any_recovery = 0
    runtime_complete_recovery = 0
    runtime_recovered_facts = 0
    # Expected-bridge offline diagnostic
    expected_bridge_skipped = 0
    expected_bridge_missing = 0
    expected_query_valid = 0
    expected_query_invalid = 0
    expected_retrieval_attempted = 0
    expected_any_recovery = 0
    expected_complete_recovery = 0
    expected_recovered_facts = 0
    # --------------------------------------------------------
    runtime_policy_counters = make_policy_counters()
    expected_policy_counters = make_policy_counters()

    # Same first 10 oracle evidence-coverage failures
    # --------------------------------------------------------
    print(
        "\nSearching for the same real "
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
        first_hop_results = reranker.search(
            query=example.question,
            embedder=embedder,
            top_k=FINAL_K,
        )
        first_hop_evidence = (
            get_retrieved_evidence(
                first_hop_results
            )
        )
        if gold.issubset(
            first_hop_evidence
        ):
            continue
        # ----------------------------------------------------
        # Offline oracle selection ONLY
        # ----------------------------------------------------
        diagnosis = (
            diagnose_oracle_retrieval(
                gold_evidence=gold,
                results=first_hop_results,
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
            - first_hop_evidence
        )
        inspected += 1
        total_missing_facts += len(
            missing
        )

        # Per-case state: never reuse another case's retrieval output.
        runtime_second_hop = None
        expected_second_hop = None
        # ----------------------------------------------------
        # Runtime-visible candidate generation
        # ----------------------------------------------------
        entities = (
            entity_extractor
            .extract_from_results(
                first_hop_results
            )
        )
        bridge_candidates = (
            build_bridge_candidates(
                entities=entities,
                results=first_hop_results,
            )
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
        print(
            "\nOriginally missing gold facts "
            "(offline evaluation only):"
        )
        for fact in sorted(
            missing
        ):
            print(
                f"  {fact}"
            )
        print(
            "\nFirst-hop final result count: "
            f"{len(first_hop_results)}"
        )
        print(
            "Runtime bridge candidate count: "
            f"{len(bridge_candidates)}"
        )
        # ====================================================
        # CONDITION A — ACTUAL RUNTIME PATH
        # ====================================================
        print()
        print(
            "-" * 100
        )
        print(
            "CONDITION A — RUNTIME SECOND-HOP "
            "RETRIEVAL"
        )
        print(
            "-" * 100
        )
        if not bridge_candidates:
            runtime_query_invalid += 1
            print(
                "Runtime path: FAILED"
            )
            print(
                "Reason: empty bridge candidate pool."
            )
        else:
            try:
                runtime_selection = (
                    llm_bridge_selector.select(
                        question=example.question,
                        candidates=(
                            bridge_candidates
                        ),
                    )
                )
                runtime_candidate = (
                    runtime_selection
                    .selected_candidate
                )
                print(
                    "Runtime selected bridge: "
                    f"{runtime_candidate.entity!r}"
                )
                runtime_query_result = (
                    relation_query_generator
                    .generate(
                        question=(
                            example.question
                        ),
                        selected_candidate=(
                            runtime_candidate
                        ),
                        candidates=(
                            bridge_candidates
                        ),
                    )
                )
            except ValueError as exc:
                runtime_query_invalid += 1
                print(
                    "Runtime query: INVALID"
                )
                print(
                    f"Reason: {exc}"
                )
            else:
                runtime_query_valid += 1
                print(
                    "Runtime query: VALID"
                )
                print(
                    "Runtime second-hop query: "
                    f"{runtime_query_result.query!r}"
                )
                # --------------------------------------------
                # Real second-hop retrieval
                #
                # No gold information is supplied here.
                # --------------------------------------------
                runtime_second_hop = (
                    second_hop_retriever.retrieve(
                        query=(
                            runtime_query_result.query
                        ),
                        first_hop_results=(
                            first_hop_results
                        ),
                    )
                )
                runtime_retrieval_attempted += 1
                # Verify preservation contract.
                assert (
                    runtime_second_hop
                    .first_hop_results
                    == tuple(
                        first_hop_results
                    )
                )
                print(
                    "\nTop-5 second-hop results:"
                )
                print_second_hop_results(
                    runtime_second_hop
                    .second_hop_results,
                    limit=5,
                )
                (
                    recovered_count,
                    complete,
                ) = print_recovery_evaluation(
                    missing=missing,
                    second_hop_results=(
                        runtime_second_hop
                        .second_hop_results
                    ),
                )
                runtime_recovered_facts += (
                    recovered_count
                )
                if recovered_count > 0:
                    runtime_any_recovery += 1
                if complete:
                    runtime_complete_recovery += 1
        # Every runtime case is evaluated, including fallback when
        # bridge/query generation fails. All policies share the same
        # second-hop result set for this case.
        evaluate_selection_policies(
            condition=(
                "Runtime" if runtime_second_hop is not None
                else "Runtime (no valid second hop)"
            ),
            gold=gold,
            first_hop_results=first_hop_results,
            second_hop_results=(
                runtime_second_hop.second_hop_results
                if runtime_second_hop is not None
                else ()
            ),
            counters=runtime_policy_counters,
        )

        # ====================================================
        # CONDITION B — OFFLINE EXPECTED-BRIDGE DIAGNOSTIC
        # ====================================================
        print()
        print(
            "-" * 100
        )
        print(
            "CONDITION B — EXPECTED-BRIDGE "
            "SECOND-HOP DIAGNOSTIC"
        )
        print(
            "-" * 100
        )
        expected_bridge = (
            EXPECTED_BRIDGES.get(
                inspected
            )
        )
        if expected_bridge is None:
            expected_bridge_skipped += 1
            print(
                "Expected-bridge diagnostic: SKIPPED"
            )
            print(
                "Reason: no clean expected bridge "
                "intervention is defined for this case."
            )
        else:
            expected_candidate = (
                find_candidate_by_entity(
                    candidates=(
                        bridge_candidates
                    ),
                    entity_name=(
                        expected_bridge
                    ),
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
            else:
                print(
                    "Expected bridge: "
                    f"{expected_candidate.entity!r}"
                )
                try:
                    expected_query_result = (
                        relation_query_generator
                        .generate(
                            question=(
                                example.question
                            ),
                            selected_candidate=(
                                expected_candidate
                            ),
                            candidates=(
                                bridge_candidates
                            ),
                        )
                    )
                except ValueError as exc:
                    expected_query_invalid += 1
                    print(
                        "Expected-bridge query: INVALID"
                    )
                    print(
                        f"Reason: {exc}"
                    )
                else:
                    expected_query_valid += 1
                    print(
                        "Expected-bridge query: VALID"
                    )
                    print(
                        "Expected-bridge second-hop "
                        "query: "
                        f"{expected_query_result.query!r}"
                    )
                    # ----------------------------------------
                    # Real second-hop retrieval.
                    #
                    # The expected bridge influenced query
                    # formulation, but gold evidence itself
                    # is NOT supplied to retrieval.
                    # ----------------------------------------
                    expected_second_hop = (
                        second_hop_retriever.retrieve(
                            query=(
                                expected_query_result.query
                            ),
                            first_hop_results=(
                                first_hop_results
                            ),
                        )
                    )
                    expected_retrieval_attempted += 1
                    assert (
                        expected_second_hop
                        .first_hop_results
                        == tuple(
                            first_hop_results
                        )
                    )
                    print(
                        "\nTop-5 second-hop results:"
                    )
                    print_second_hop_results(
                        expected_second_hop
                        .second_hop_results,
                        limit=5,
                    )
                    (
                        recovered_count,
                        complete,
                    ) = print_recovery_evaluation(
                        missing=missing,
                        second_hop_results=(
                            expected_second_hop
                            .second_hop_results
                        ),
                    )
                    expected_recovered_facts += (
                        recovered_count
                    )
                    if recovered_count > 0:
                        expected_any_recovery += 1
                    if complete:
                        (
                            expected_complete_recovery
                        ) += 1
        # Offline expected-bridge intervention: include only cases
        # where a valid query led to actual second-hop retrieval.
        if expected_second_hop is not None:
            evaluate_selection_policies(
                condition="Expected bridge (offline diagnostic)",
                gold=gold,
                first_hop_results=first_hop_results,
                second_hop_results=expected_second_hop.second_hop_results,
                counters=expected_policy_counters,
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
        "PHASE 10.4C SECOND-HOP "
        "RETRIEVAL SUMMARY"
    )
    print(
        "=" * 100
    )
    print(
        "\nCoverage failures inspected: "
        f"{inspected}"
    )
    print(
        "Originally missing facts: "
        f"{total_missing_facts}"
    )
    print(
        "\nCondition A — runtime path:"
    )
    print(
        "  Valid second-hop queries: "
        f"{runtime_query_valid}"
    )
    print(
        "  Invalid/unavailable queries: "
        f"{runtime_query_invalid}"
    )
    print(
        "  Retrieval attempts: "
        f"{runtime_retrieval_attempted}"
    )
    print(
        "  Cases with any missing evidence "
        "recovered: "
        f"{runtime_any_recovery}"
    )
    print(
        "  Cases with all missing evidence "
        "recovered: "
        f"{runtime_complete_recovery}"
    )
    print(
        "  Missing facts recovered: "
        f"{runtime_recovered_facts}"
    )
    print(
        "\nCondition B — expected-bridge "
        "offline diagnostic:"
    )
    print(
        "  Valid second-hop queries: "
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
    print(
        "  Retrieval attempts: "
        f"{expected_retrieval_attempted}"
    )
    print(
        "  Cases with any missing evidence "
        "recovered: "
        f"{expected_any_recovery}"
    )
    print(
        "  Cases with all missing evidence "
        "recovered: "
        f"{expected_complete_recovery}"
    )
    print(
        "  Missing facts recovered: "
        f"{expected_recovered_facts}"
    )
    # --------------------------------------------------------
    # Structural checks
    # --------------------------------------------------------
    assert inspected == MAX_CASES
    assert (
        runtime_query_valid
        + runtime_query_invalid
        == inspected
    )
    assert (
        runtime_retrieval_attempted
        == runtime_query_valid
    )
    assert (
        expected_retrieval_attempted
        == expected_query_valid
    )
    assert (
        expected_query_valid
        + expected_query_invalid
        + expected_bridge_missing
        + expected_bridge_skipped
        == inspected
    )
    # --------------------------------------------------------
    # Methodological guarantees
    # --------------------------------------------------------
    print(
        "\nPASS: The same frozen first-hop "
        "retrieval configuration was preserved."
    )
    print(
        "PASS: Runtime second-hop retrieval "
        "received only the generated query and "
        "the frozen retrieval stack."
    )
    print(
        "PASS: Gold evidence was used only after "
        "retrieval for offline recovery scoring."
    )
    print(
        "PASS: Recovery was scored only against "
        "facts missing from the original first-hop "
        "final top-k."
    )
    print(
        "PASS: First-hop and second-hop results "
        "were preserved separately and were not "
        "implicitly merged."
    )
    print(
        "PASS: Condition B remained an offline "
        "expected-bridge intervention and is not "
        "reported as runtime recovery."
    )
    print_policy_summary(
        "CONDITION A — RUNTIME",
        runtime_policy_counters,
    )
    print_policy_summary(
        "CONDITION B — EXPECTED-BRIDGE DIAGNOSTIC",
        expected_policy_counters,
    )

    assert all(
        stats["evaluated"] == MAX_CASES
        for stats in runtime_policy_counters.values()
    ), "Each runtime policy must evaluate every selected case."

    assert all(
        stats["evaluated"] == expected_retrieval_attempted
        for stats in expected_policy_counters.values()
    ), "Expected-bridge policies must evaluate only retrieval attempts."

    assert all(
        runtime_policy_counters["First-hop only"][key]
        == runtime_policy_counters["Naive rank union"][key]
        for key in (
            "gold_total", "gold_retrieved", "complete",
            "recovered", "lost", "repaired_cases", "harmed_cases",
        )
    ), "Naive first-hop-priority union must match first-hop-only at k=10."

    print(
        "\nNOTE: Gold evidence is used only for offline scoring. "
        "The ten cases are oracle-selected validation diagnostics, "
        "not an unbiased production recovery estimate."
    )
    print("\nPHASE 10.4C REAL SELECTION COMPARISON COMPLETED.")
if __name__ == "__main__":
    main()
