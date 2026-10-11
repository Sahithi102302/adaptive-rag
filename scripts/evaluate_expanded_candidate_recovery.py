from __future__ import annotations

from collections import Counter

import numpy as np

from src.diagnosis.oracle import diagnose_oracle_retrieval
from src.diagnosis.states import FailureType
from src.embeddings.dense_embedder import (
    DenseEmbedder,
    DenseEmbeddingConfig,
)
from src.retrieval.dense_retriever import (
    DenseRetriever,
    format_chunk_for_retrieval,
)
from src.retrieval.reranker import (
    CrossEncoderReranker,
    RerankerConfig,
    RerankedRetrievalResult,
)
from src.retrieval.sparse_retriever import BM25Retriever

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


# ============================================================
# Configuration
# ============================================================

SPLIT_NAME = "validation"

BASELINE_CANDIDATE_K = 20

CANDIDATE_DEPTHS = (
    20,
    40,
    60,
    100,
)

MAX_CANDIDATE_K = max(
    CANDIDATE_DEPTHS
)

FINAL_K = 10

BATCH_SIZE = 32


# ============================================================
# Evidence helpers
# ============================================================


def get_chunk_evidence(
    chunk,
) -> set[tuple[str, int]]:
    """
    Return the HotpotQA evidence represented by one
    sentence-aware chunk.
    """

    return {
        (
            chunk.document_id,
            sentence_id,
        )
        for sentence_id in chunk.metadata.get(
            "sentence_ids",
            [],
        )
    }


def get_retrieved_evidence(
    results: list[RerankedRetrievalResult],
) -> set[tuple[str, int]]:
    """
    Aggregate evidence represented by retrieved chunks.
    """

    evidence: set[
        tuple[str, int]
    ] = set()

    for result in results:
        evidence.update(
            get_chunk_evidence(
                result.chunk
            )
        )

    return evidence


def is_complete(
    gold: set[tuple[str, int]],
    results: list[RerankedRetrievalResult],
) -> bool:
    """
    Whether all valid gold evidence is represented.
    """

    retrieved = get_retrieved_evidence(
        results
    )

    return gold.issubset(
        retrieved
    )


# ============================================================
# Candidate construction
# ============================================================


def build_candidates(
    dense_results,
    sparse_results,
) -> dict[str, dict]:
    """
    Build the union of dense and sparse candidates while
    retaining each candidate's rank in both retrievers.
    """

    candidates: dict[
        str,
        dict,
    ] = {}

    for result in dense_results:

        chunk_id = result.chunk.id

        if chunk_id not in candidates:
            candidates[chunk_id] = {
                "chunk": result.chunk,
                "dense_rank": None,
                "sparse_rank": None,
            }

        candidates[
            chunk_id
        ]["dense_rank"] = result.rank

    for result in sparse_results:

        chunk_id = result.chunk.id

        if chunk_id not in candidates:
            candidates[chunk_id] = {
                "chunk": result.chunk,
                "dense_rank": None,
                "sparse_rank": None,
            }

        candidates[
            chunk_id
        ]["sparse_rank"] = result.rank

    return candidates


def candidate_is_available_at_depth(
    candidate: dict,
    candidate_k: int,
) -> bool:
    """
    Whether the candidate would have entered the dense/BM25
    union if both retrievers had been truncated at candidate_k.
    """

    dense_rank = candidate[
        "dense_rank"
    ]

    sparse_rank = candidate[
        "sparse_rank"
    ]

    return (
        (
            dense_rank is not None
            and dense_rank <= candidate_k
        )
        or (
            sparse_rank is not None
            and sparse_rank <= candidate_k
        )
    )


def rerank_for_depth(
    candidates: dict[str, dict],
    scores_by_chunk_id: dict[str, float],
    candidate_k: int,
    final_k: int,
) -> list[RerankedRetrievalResult]:
    """
    Reconstruct the exact CE ranking that would result from
    retrieving dense@candidate_k and BM25@candidate_k.

    CE scores are pointwise, so candidates can be scored once
    at the maximum depth and filtered for smaller depths.
    """

    available = [
        candidate
        for candidate in candidates.values()
        if candidate_is_available_at_depth(
            candidate,
            candidate_k,
        )
    ]

    available.sort(
        key=lambda candidate: (
            -scores_by_chunk_id[
                candidate["chunk"].id
            ],
            candidate["chunk"].id,
        )
    )

    results: list[
        RerankedRetrievalResult
    ] = []

    for rank, candidate in enumerate(
        available[:final_k],
        start=1,
    ):

        chunk = candidate[
            "chunk"
        ]

        results.append(
            RerankedRetrievalResult(
                rank=rank,
                score=scores_by_chunk_id[
                    chunk.id
                ],
                chunk=chunk,
                dense_rank=(
                    candidate[
                        "dense_rank"
                    ]
                ),
                sparse_rank=(
                    candidate[
                        "sparse_rank"
                    ]
                ),
            )
        )

    return results


# ============================================================
# Main
# ============================================================


def main() -> None:

    print("=" * 100)
    print(
        "PHASE 10.2B — EXPANDED CANDIDATE "
        "RECOVERY EVALUATION"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # Frozen validation queries
    # --------------------------------------------------------

    print(
        "\nLoading frozen validation queries..."
    )

    examples = load_query_examples(
        SPLIT_NAME
    )

    assert len(examples) == 200

    print(
        f"Validation questions: "
        f"{len(examples):,}"
    )

    # --------------------------------------------------------
    # Frozen retrieval corpus
    # --------------------------------------------------------

    print(
        "\nLoading frozen chunks..."
    )

    chunks = load_frozen_chunks()

    assert len(chunks) == EXPECTED_CHUNKS

    print(
        f"Frozen chunks: "
        f"{len(chunks):,}"
    )

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
    # Retrieval infrastructure
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

    assert (
        embedder.embedding_dimension
        == EXPECTED_DIMENSION
    )

    # --------------------------------------------------------
    # Cross-encoder
    #
    # We instantiate the existing class only so we reuse the
    # exact same CE model/configuration as Phase 8/9.
    # Candidate construction below is performed explicitly.
    # --------------------------------------------------------

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
            candidate_k=MAX_CANDIDATE_K,
            batch_size=BATCH_SIZE,
        ),
    )

    # --------------------------------------------------------
    # Accumulators
    # --------------------------------------------------------

    total_gold_facts = 0

    evidence_hits = {
        depth: 0
        for depth in CANDIDATE_DEPTHS
    }

    complete_questions = {
        depth: 0
        for depth in CANDIDATE_DEPTHS
    }

    question_transitions = {
        depth: Counter()
        for depth in CANDIDATE_DEPTHS
        if depth != BASELINE_CANDIDATE_K
    }

    fact_gains = {
        depth: 0
        for depth in CANDIDATE_DEPTHS
        if depth != BASELINE_CANDIDATE_K
    }

    fact_losses = {
        depth: 0
        for depth in CANDIDATE_DEPTHS
        if depth != BASELINE_CANDIDATE_K
    }

    oracle_failure_repairs = {
        depth: Counter()
        for depth in CANDIDATE_DEPTHS
        if depth != BASELINE_CANDIDATE_K
    }

    oracle_failure_totals = Counter()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print(
        "\nEvaluating candidate depths "
        f"{CANDIDATE_DEPTHS}...\n"
    )

    for index, example in enumerate(
        examples,
        start=1,
    ):

        gold = get_gold_evidence(
            example
        )

        if not gold:
            raise RuntimeError(
                "Validation example has zero "
                f"valid gold facts at position {index}."
            )

        total_gold_facts += len(
            gold
        )

        # ----------------------------------------------------
        # Retrieve maximum depth once
        # ----------------------------------------------------

        dense_results = (
            dense_retriever.search(
                query=example.question,
                embedder=embedder,
                top_k=MAX_CANDIDATE_K,
            )
        )

        sparse_results = (
            sparse_retriever.search(
                query=example.question,
                top_k=MAX_CANDIDATE_K,
            )
        )

        candidates = build_candidates(
            dense_results=dense_results,
            sparse_results=sparse_results,
        )

        # ----------------------------------------------------
        # Score candidate union once
        # ----------------------------------------------------

        candidate_list = list(
            candidates.values()
        )

        pairs = [
            (
                example.question,
                format_chunk_for_retrieval(
                    candidate["chunk"]
                ),
            )
            for candidate in candidate_list
        ]

        scores = reranker.model.predict(
            pairs,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
        )

        scores_by_chunk_id = {
            candidate["chunk"].id: float(
                score
            )
            for candidate, score in zip(
                candidate_list,
                scores,
            )
        }

        # ----------------------------------------------------
        # Reconstruct each candidate-depth experiment
        # ----------------------------------------------------

        results_by_depth = {}

        retrieved_gold_by_depth = {}

        complete_by_depth = {}

        for depth in CANDIDATE_DEPTHS:

            results = rerank_for_depth(
                candidates=candidates,
                scores_by_chunk_id=(
                    scores_by_chunk_id
                ),
                candidate_k=depth,
                final_k=FINAL_K,
            )

            if len(results) != FINAL_K:
                raise RuntimeError(
                    "Unexpected number of results "
                    f"for candidate_k={depth}, "
                    f"question={index}: "
                    f"{len(results)}"
                )

            results_by_depth[
                depth
            ] = results

            retrieved_evidence = (
                get_retrieved_evidence(
                    results
                )
            )

            retrieved_gold = (
                gold
                & retrieved_evidence
            )

            retrieved_gold_by_depth[
                depth
            ] = retrieved_gold

            evidence_hits[
                depth
            ] += len(
                retrieved_gold
            )

            complete = gold.issubset(
                retrieved_evidence
            )

            complete_by_depth[
                depth
            ] = complete

            if complete:
                complete_questions[
                    depth
                ] += 1

        # ----------------------------------------------------
        # Baseline oracle failure type
        # ----------------------------------------------------

        baseline_results = (
            results_by_depth[
                BASELINE_CANDIDATE_K
            ]
        )

        baseline_diagnosis = (
            diagnose_oracle_retrieval(
                gold_evidence=gold,
                results=baseline_results,
            )
        )

        baseline_complete = (
            complete_by_depth[
                BASELINE_CANDIDATE_K
            ]
        )

        if not baseline_complete:
            oracle_failure_totals[
                baseline_diagnosis
                .failure_type
                .value
            ] += 1

        baseline_gold = (
            retrieved_gold_by_depth[
                BASELINE_CANDIDATE_K
            ]
        )

        # ----------------------------------------------------
        # Before → after analysis
        # ----------------------------------------------------

        for depth in CANDIDATE_DEPTHS:

            if depth == BASELINE_CANDIDATE_K:
                continue

            recovered_complete = (
                complete_by_depth[
                    depth
                ]
            )

            if (
                not baseline_complete
                and recovered_complete
            ):
                transition = "repaired"

            elif (
                not baseline_complete
                and not recovered_complete
            ):
                transition = "unrepaired"

            elif (
                baseline_complete
                and recovered_complete
            ):
                transition = "preserved"

            else:
                transition = "harmed"

            question_transitions[
                depth
            ][transition] += 1

            recovered_gold = (
                retrieved_gold_by_depth[
                    depth
                ]
            )

            fact_gains[
                depth
            ] += len(
                recovered_gold
                - baseline_gold
            )

            fact_losses[
                depth
            ] += len(
                baseline_gold
                - recovered_gold
            )

            if (
                not baseline_complete
                and recovered_complete
            ):
                oracle_failure_repairs[
                    depth
                ][
                    baseline_diagnosis
                    .failure_type
                    .value
                ] += 1

        if (
            index % 10 == 0
            or index == len(examples)
        ):
            print(
                f"Processed "
                f"{index}/"
                f"{len(examples)}"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total_questions = len(
        examples
    )

    baseline_complete_count = (
        complete_questions[
            BASELINE_CANDIDATE_K
        ]
    )

    baseline_insufficient_count = (
        total_questions
        - baseline_complete_count
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "EXPANDED-CANDIDATE RECOVERY RESULTS"
    )

    print("=" * 100)

    print(
        f"\nQuestions: "
        f"{total_questions}"
    )

    print(
        f"Valid gold facts: "
        f"{total_gold_facts}"
    )

    print(
        f"\nBaseline candidate_k: "
        f"{BASELINE_CANDIDATE_K}"
    )

    print(
        f"Baseline sufficient: "
        f"{baseline_complete_count}"
    )

    print(
        f"Baseline insufficient: "
        f"{baseline_insufficient_count}"
    )

    print(
        "\nRetrieval quality by candidate depth:"
    )

    for depth in CANDIDATE_DEPTHS:

        recall = (
            evidence_hits[depth]
            / total_gold_facts
        )

        complete_rate = (
            complete_questions[depth]
            / total_questions
        )

        print(
            f"\n  candidate_k={depth}"
        )

        print(
            "    Evidence Recall@10: "
            f"{recall:.4f} "
            f"({evidence_hits[depth]}/"
            f"{total_gold_facts})"
        )

        print(
            "    Complete Evidence@10: "
            f"{complete_rate:.4f} "
            f"({complete_questions[depth]}/"
            f"{total_questions})"
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "RECOVERY TRANSITIONS RELATIVE TO "
        "candidate_k=20"
    )

    print("=" * 100)

    for depth in CANDIDATE_DEPTHS:

        if depth == BASELINE_CANDIDATE_K:
            continue

        transitions = (
            question_transitions[
                depth
            ]
        )

        repaired = transitions[
            "repaired"
        ]

        unrepaired = transitions[
            "unrepaired"
        ]

        preserved = transitions[
            "preserved"
        ]

        harmed = transitions[
            "harmed"
        ]

        repair_rate = (
            repaired
            / baseline_insufficient_count
            if baseline_insufficient_count
            else 0.0
        )

        harm_rate = (
            harmed
            / baseline_complete_count
            if baseline_complete_count
            else 0.0
        )

        print(
            f"\ncandidate_k={depth}"
        )

        print(
            f"  repaired:   "
            f"{repaired}/"
            f"{baseline_insufficient_count} "
            f"({repair_rate:.4f})"
        )

        print(
            f"  unrepaired: "
            f"{unrepaired}/"
            f"{baseline_insufficient_count}"
        )

        print(
            f"  preserved:  "
            f"{preserved}/"
            f"{baseline_complete_count}"
        )

        print(
            f"  harmed:     "
            f"{harmed}/"
            f"{baseline_complete_count} "
            f"({harm_rate:.4f})"
        )

        print(
            f"  gold facts gained: "
            f"{fact_gains[depth]}"
        )

        print(
            f"  gold facts lost:   "
            f"{fact_losses[depth]}"
        )

    # --------------------------------------------------------
    # Oracle failure-type repair analysis
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "ORACLE FAILURE-TYPE REPAIR ANALYSIS"
    )

    print("=" * 100)

    print(
        "\nBaseline failure distribution:"
    )

    for failure_type, count in (
        oracle_failure_totals.items()
    ):
        print(
            f"  {failure_type}: "
            f"{count}"
        )

    for depth in CANDIDATE_DEPTHS:

        if depth == BASELINE_CANDIDATE_K:
            continue

        print(
            f"\ncandidate_k={depth}"
        )

        for (
            failure_type,
            total,
        ) in oracle_failure_totals.items():

            repaired = (
                oracle_failure_repairs[
                    depth
                ][failure_type]
            )

            rate = (
                repaired / total
                if total
                else 0.0
            )

            print(
                f"  {failure_type}: "
                f"{repaired}/{total} "
                f"({rate:.4f})"
            )

    # --------------------------------------------------------
    # Consistency checks
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "CONSISTENCY CHECKS"
    )

    print("=" * 100)

    for depth in CANDIDATE_DEPTHS:

        if depth == BASELINE_CANDIDATE_K:
            continue

        transitions = (
            question_transitions[
                depth
            ]
        )

        assert (
            transitions["repaired"]
            + transitions["unrepaired"]
            == baseline_insufficient_count
        )

        assert (
            transitions["preserved"]
            + transitions["harmed"]
            == baseline_complete_count
        )

        assert sum(
            transitions.values()
        ) == total_questions

    assert (
        sum(
            oracle_failure_totals.values()
        )
        == baseline_insufficient_count
    )

    print(
        "\nPASS: Every validation question "
        "has exactly one before/after transition."
    )

    print(
        "PASS: Oracle failure counts match "
        "baseline insufficient questions."
    )

    print(
        "PASS: Gold evidence was used only "
        "for offline recovery evaluation."
    )

    print(
        "\nNOTE: Candidate depth has NOT been "
        "selected or frozen by this experiment."
    )

    print(
        "NOTE: These are validation results, "
        "not held-out test results."
    )


if __name__ == "__main__":
    main()