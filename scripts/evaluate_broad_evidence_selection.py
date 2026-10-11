"""Phase 10.4E: broader, paired validation of final-context selection.

Runs a diagnostic recovery attempt on EVERY sampled validation question,
including initially complete cases. This is deliberately NOT the Phase 9
sufficiency-gated production policy. Gold is accessed only for offline metrics.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from scripts.build_scaled_diagnostic_dataset import (
    CROSS_ENCODER_MODEL_NAME, DENSE_MODEL_NAME, EMBEDDINGS_PATH,
    EXPECTED_CHUNKS, EXPECTED_DIMENSION, get_gold_evidence,
    load_frozen_chunks, load_query_examples,
)
from scripts.evaluate_expanded_candidate_recovery import get_retrieved_evidence
from scripts.evaluate_real_evidence_selection import (
    BASELINE_CANDIDATE_K, BATCH_SIZE, ENTITY_CONFIG, FINAL_K,
    LLM_BRIDGE_SELECTION_CONFIG, QWEN_CONFIG, RELATION_QUERY_CONFIG,
    SECOND_HOP_CONFIG, build_bridge_candidates, select_policy_results,
)
from src.embeddings.dense_embedder import DenseEmbedder, DenseEmbeddingConfig
from src.recovery.entity_extraction import ContentEntityExtractor
from src.recovery.llm_bridge_selection import ClosedCandidateBridgeSelector
from src.recovery.qwen_generator import QwenTextGenerator
from src.recovery.relation_query import ConstrainedRelationQueryGenerator
from src.recovery.second_hop import SecondHopRetriever
from src.retrieval.dense_retriever import DenseRetriever, format_chunk_for_retrieval
from src.retrieval.reranker import CrossEncoderReranker, RerankerConfig, RerankedRetrievalResult
from src.retrieval.sparse_retriever import BM25Retriever

POLICIES = (
    "First-hop only", "Naive rank union", "Hop-balanced 5/5",
    "Hop-balanced 7/3", "Hop-balanced 3/7", "Combined cross-encoder",
)


def combined_cross_encoder(question, first, second, reranker):
    """Score the deduplicated top-10 + top-10 pool against ORIGINAL question."""
    unique = {}
    for result in (*first, *second):
        unique.setdefault(result.chunk.id, result)
    candidates = list(unique.values())
    if not candidates:
        return ()
    pairs = [(question, format_chunk_for_retrieval(r.chunk)) for r in candidates]
    scores = reranker.model.predict(
        pairs, batch_size=reranker.config.batch_size, show_progress_bar=False
    )
    if len(scores) != len(candidates):
        raise AssertionError("Cross-encoder returned an unexpected number of scores")
    scored = sorted(
        zip(scores, candidates),
        key=lambda pair: (-float(pair[0]), pair[1].chunk.id),
    )
    return tuple(
        RerankedRetrievalResult(
            rank=rank, score=float(score), chunk=original.chunk,
            dense_rank=original.dense_rank, sparse_rank=original.sparse_rank,
        )
        for rank, (score, original) in enumerate(scored[:FINAL_K], 1)
    )


def create_stack():
    print("Loading frozen validation corpus and embeddings...", flush=True)
    chunks = load_frozen_chunks()
    assert len(chunks) == EXPECTED_CHUNKS
    embeddings = np.load(EMBEDDINGS_PATH)
    assert embeddings.shape == (EXPECTED_CHUNKS, EXPECTED_DIMENSION)
    assert embeddings.dtype == np.float32
    dense = DenseRetriever(chunks=chunks, embeddings=embeddings)
    sparse = BM25Retriever(
        chunks=chunks,
        retrieval_texts=[format_chunk_for_retrieval(chunk) for chunk in chunks],
    )
    embedder = DenseEmbedder(DenseEmbeddingConfig(
        model_name=DENSE_MODEL_NAME, batch_size=BATCH_SIZE,
        normalize_embeddings=True,
    ))
    reranker = CrossEncoderReranker(
        dense_retriever=dense, sparse_retriever=sparse,
        config=RerankerConfig(
            model_name=CROSS_ENCODER_MODEL_NAME,
            candidate_k=BASELINE_CANDIDATE_K, batch_size=BATCH_SIZE,
        ),
    )
    second_hop = SecondHopRetriever(
        reranker=reranker, embedder=embedder, config=SECOND_HOP_CONFIG
    )
    extractor = ContentEntityExtractor(config=ENTITY_CONFIG)
    generator = QwenTextGenerator(config=QWEN_CONFIG)
    selector = ClosedCandidateBridgeSelector(
        generator=generator, config=LLM_BRIDGE_SELECTION_CONFIG
    )
    query_generator = ConstrainedRelationQueryGenerator(
        generator=generator, config=RELATION_QUERY_CONFIG
    )
    return embedder, reranker, second_hop, extractor, selector, query_generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-questions", type=int, default=50,
                        help="Number of validation positions to inspect (default: 50; max: 200)")
    parser.add_argument("--output", default="docs/experiments/phase_10_4e_results.csv")
    args = parser.parse_args()
    if not 1 <= args.max_questions <= 200:
        parser.error("--max-questions must be between 1 and 200")

    examples = load_query_examples("validation")
    assert len(examples) == 200
    embedder, reranker, second_hop, extractor, bridge_selector, query_generator = create_stack()
    rows = []
    processed = 0
    print("\nPHASE 10.4E — PAIRED BROADER VALIDATION", flush=True)
    print("Diagnostic intervention on all sampled questions, NOT a production trigger.", flush=True)

    for position, example in enumerate(examples[:args.max_questions], start=1):
        # Runtime path: question + retrieved text only; no gold consulted.
        first = tuple(reranker.search(
            query=example.question, embedder=embedder, top_k=FINAL_K
        ))
        second = ()
        status = "no_candidates"
        bridge = ""
        recovery_query = ""
        try:
            entities = extractor.extract_from_results(first)
            candidates = build_bridge_candidates(entities=entities, results=first)
            if candidates:
                selected_bridge = bridge_selector.select(
                    question=example.question, candidates=candidates
                ).selected_candidate
                bridge = selected_bridge.entity
                generated = query_generator.generate(
                    question=example.question,
                    selected_candidate=selected_bridge,
                    candidates=candidates,
                )
                recovery_query = generated.query
                retrieval = second_hop.retrieve(
                    query=recovery_query, first_hop_results=first
                )
                assert retrieval.first_hop_results == first
                second = tuple(retrieval.second_hop_results)
                status = "retrieved"
        except ValueError as exc:
            status = "invalid_query_or_bridge"
            print(f"  position={position} recovery skipped: {exc}", flush=True)

        # Build ALL final contexts without access to gold annotations.
        selected_by_policy = {
            name: (combined_cross_encoder(example.question, first, second, reranker)
                   if name == "Combined cross-encoder"
                   else select_policy_results(name, first, second))
            for name in POLICIES
        }
        for name, selected in selected_by_policy.items():
            ids = [result.chunk.id for result in selected]
            assert len(ids) <= FINAL_K and len(ids) == len(set(ids)), name
        assert [r.chunk.id for r in selected_by_policy["First-hop only"]] == [
            r.chunk.id for r in selected_by_policy["Naive rank union"]
        ], "First-hop-priority naive union must match first-hop top-10"
        if not second:
            assert all(
                {r.chunk.id for r in selected} == {r.chunk.id for r in first}
                for selected in selected_by_policy.values()
            ), "No valid recovery should preserve first-hop context"

        # OFFLINE SCORING ONLY: gold first accessed after ALL selection.
        gold = get_gold_evidence(example)
        if not gold:
            print(f"  position={position}: no gold evidence, skipped", flush=True)
            continue
        original = get_retrieved_evidence(first) & gold
        originally_complete = gold.issubset(original)
        for policy, selected in selected_by_policy.items():
            final = get_retrieved_evidence(selected) & gold
            recovered = (gold - original) & final
            lost = original - final
            rows.append({
                "position": position, "policy": policy, "status": status,
                "originally_complete": int(originally_complete),
                "gold_total": len(gold), "original_gold": len(original),
                "final_gold": len(final), "recovered": len(recovered),
                "lost": len(lost), "final_complete": int(gold.issubset(final)),
                "repaired": int(not originally_complete and gold.issubset(final)),
                "harmed": int(bool(lost)),
                "complete_to_incomplete": int(originally_complete and not gold.issubset(final)),
                "second_hop_count": len(second),
                "bridge": bridge, "recovery_query": recovery_query,
            })
        processed += 1
        print(
            f"[{position}/{args.max_questions}] "
            f"original_complete={originally_complete} "
            f"second_hop={status} "
            f"balanced_5_5_complete={next(r['final_complete'] for r in rows[-len(POLICIES):] if r['policy']=='Hop-balanced 5/5')}",
            flush=True,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No evaluable cases")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nEvaluated {processed} validation questions; output: {output}")
    for policy in POLICIES:
        subset = [r for r in rows if r["policy"] == policy]
        total = sum(r["gold_total"] for r in subset)
        recall = sum(r["final_gold"] for r in subset) / total
        print(
            f"{policy:25s} recall={recall:.4f} "
            f"complete={sum(r['final_complete'] for r in subset)}/{len(subset)} "
            f"recovered={sum(r['recovered'] for r in subset)} "
            f"lost={sum(r['lost'] for r in subset)} "
            f"repaired={sum(r['repaired'] for r in subset)} "
            f"harmed_cases={sum(r['harmed'] for r in subset)} "
            f"complete_to_incomplete={sum(r['complete_to_incomplete'] for r in subset)}"
        )
    print("\nNOTE: This diagnostic forces recovery attempts on complete and incomplete cases.")
    print("No oracle labels or gold evidence were used to trigger recovery or select passages.")
    print("PHASE 10.4E COMPLETED.")


if __name__ == "__main__":
    main()
