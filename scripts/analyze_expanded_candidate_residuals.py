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
    build_candidates,
    get_chunk_evidence,
    get_retrieved_evidence,
    rerank_for_depth,
)


# ============================================================
# Configuration
# ============================================================

SPLIT_NAME = "validation"

BASELINE_CANDIDATE_K = 20
EXPANDED_CANDIDATE_K = 100

FINAL_K = 10

BATCH_SIZE = 32

SAMPLE_LIMIT = 10


# ============================================================
# Gold candidate analysis
# ============================================================


def candidate_gold_evidence(
    candidate: dict,
    gold: set[tuple[str, int]],
) -> set[tuple[str, int]]:
    """
    Gold evidence represented by one candidate chunk.
    """

    return (
        get_chunk_evidence(
            candidate["chunk"]
        )
        & gold
    )


def find_gold_candidate_locations(
    candidates: dict[str, dict],
    gold: set[tuple[str, int]],
) -> dict[
    tuple[str, int],
    list[dict],
]:
    """
    For every gold fact, find candidate chunks within the
    dense@100 / BM25@100 union that contain that fact.
    """

    locations: dict[
        tuple[str, int],
        list[dict],
    ] = {
        fact: []
        for fact in gold
    }

    for candidate in candidates.values():

        matched = candidate_gold_evidence(
            candidate,
            gold,
        )

        for fact in matched:
            locations[fact].append(
                candidate
            )

    return locations


def classify_missing_fact(
    fact: tuple[str, int],
    locations: dict[
        tuple[str, int],
        list[dict],
    ],
    expanded_results,
) -> str:
    """
    Classify why a gold fact remains absent from final top-10.

    unavailable_at_100:
        No chunk containing the fact appeared in the
        dense@100/BM25@100 union.

    available_but_not_top10:
        The fact was available to the cross-encoder but all
        chunks containing it were ranked outside final top-10.

    unexpected:
        Defensive category. If the fact is represented in
        final top-10 it should not have been considered missing.
    """

    fact_locations = locations[
        fact
    ]

    if not fact_locations:
        return "unavailable_at_100"

    final_evidence = (
        get_retrieved_evidence(
            expanded_results
        )
    )

    if fact not in final_evidence:
        return "available_but_not_top10"

    return "unexpected"


def format_rank(
    rank,
) -> str:

    if rank is None:
        return "-"

    return str(rank)


# ============================================================
# Main
# ============================================================


def main() -> None:

    print("=" * 100)
    print(
        "PHASE 10.2C — EXPANDED-CANDIDATE "
        "RESIDUAL FAILURE ANALYSIS"
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
                EXPANDED_CANDIDATE_K
            ),
            batch_size=BATCH_SIZE,
        ),
    )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    baseline_failures = 0
    residual_failures = 0

    baseline_failure_types = Counter()
    residual_failure_types = Counter()

    missing_fact_causes = Counter()

    residual_records = []

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print(
        "\nAnalyzing residual failures...\n"
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

        # Retrieve maximum candidate depth once.

        dense_results = dense_retriever.search(
            query=example.question,
            embedder=embedder,
            top_k=EXPANDED_CANDIDATE_K,
        )

        sparse_results = sparse_retriever.search(
            query=example.question,
            top_k=EXPANDED_CANDIDATE_K,
        )

        candidates = build_candidates(
            dense_results=dense_results,
            sparse_results=sparse_results,
        )

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

        # Reconstruct baseline and expanded results.

        baseline_results = rerank_for_depth(
            candidates=candidates,
            scores_by_chunk_id=(
                scores_by_chunk_id
            ),
            candidate_k=(
                BASELINE_CANDIDATE_K
            ),
            final_k=FINAL_K,
        )

        expanded_results = rerank_for_depth(
            candidates=candidates,
            scores_by_chunk_id=(
                scores_by_chunk_id
            ),
            candidate_k=(
                EXPANDED_CANDIDATE_K
            ),
            final_k=FINAL_K,
        )

        baseline_evidence = (
            get_retrieved_evidence(
                baseline_results
            )
        )

        expanded_evidence = (
            get_retrieved_evidence(
                expanded_results
            )
        )

        baseline_complete = (
            gold.issubset(
                baseline_evidence
            )
        )

        expanded_complete = (
            gold.issubset(
                expanded_evidence
            )
        )

        # We only care about original failures.

        if baseline_complete:
            if (
                index % 10 == 0
                or index == len(examples)
            ):
                print(
                    f"Processed "
                    f"{index}/"
                    f"{len(examples)}"
                )

            continue

        baseline_failures += 1

        baseline_diagnosis = (
            diagnose_oracle_retrieval(
                gold_evidence=gold,
                results=baseline_results,
            )
        )

        baseline_failure_types[
            baseline_diagnosis
            .failure_type
            .value
        ] += 1

        # Repaired cases are not residual failures.

        if expanded_complete:
            if (
                index % 10 == 0
                or index == len(examples)
            ):
                print(
                    f"Processed "
                    f"{index}/"
                    f"{len(examples)}"
                )

            continue

        residual_failures += 1

        residual_diagnosis = (
            diagnose_oracle_retrieval(
                gold_evidence=gold,
                results=expanded_results,
            )
        )

        residual_failure_types[
            residual_diagnosis
            .failure_type
            .value
        ] += 1

        missing = (
            gold
            - expanded_evidence
        )

        locations = (
            find_gold_candidate_locations(
                candidates=candidates,
                gold=gold,
            )
        )

        cause_counts = Counter()

        for fact in missing:

            cause = classify_missing_fact(
                fact=fact,
                locations=locations,
                expanded_results=(
                    expanded_results
                ),
            )

            missing_fact_causes[
                cause
            ] += 1

            cause_counts[
                cause
            ] += 1

        residual_records.append(
            {
                "index": index,
                "question": (
                    example.question
                ),
                "baseline_failure": (
                    baseline_diagnosis
                    .failure_type
                    .value
                ),
                "residual_failure": (
                    residual_diagnosis
                    .failure_type
                    .value
                ),
                "gold": gold,
                "expanded_evidence": (
                    expanded_evidence
                ),
                "missing": missing,
                "locations": locations,
                "cause_counts": (
                    cause_counts
                ),
                "expanded_results": (
                    expanded_results
                ),
            }
        )

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

    print(
        "\n"
        + "=" * 100
    )

    print(
        "RESIDUAL FAILURE SUMMARY"
    )

    print("=" * 100)

    print(
        f"\nBaseline failures: "
        f"{baseline_failures}"
    )

    print(
        f"Residual failures after "
        f"candidate_k="
        f"{EXPANDED_CANDIDATE_K}: "
        f"{residual_failures}"
    )

    print(
        f"Repaired failures: "
        f"{baseline_failures - residual_failures}"
    )

    print(
        "\nBaseline failure types:"
    )

    for failure_type, count in (
        baseline_failure_types.items()
    ):
        print(
            f"  {failure_type}: "
            f"{count}"
        )

    print(
        "\nResidual failure types:"
    )

    for failure_type, count in (
        residual_failure_types.items()
    ):
        print(
            f"  {failure_type}: "
            f"{count}"
        )

    print(
        "\nMissing gold-fact causes "
        "after candidate_k=100:"
    )

    total_missing_facts = sum(
        missing_fact_causes.values()
    )

    for cause, count in (
        missing_fact_causes.items()
    ):

        fraction = (
            count / total_missing_facts
            if total_missing_facts
            else 0.0
        )

        print(
            f"  {cause}: "
            f"{count}/"
            f"{total_missing_facts} "
            f"({fraction:.4f})"
        )

    # --------------------------------------------------------
    # Sample residual cases
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "SAMPLE RESIDUAL FAILURES"
    )

    print("=" * 100)

    for sample_number, record in enumerate(
        residual_records[
            :SAMPLE_LIMIT
        ],
        start=1,
    ):

        print(
            f"\n--- Residual "
            f"{sample_number} ---"
        )

        print(
            f"Validation position: "
            f"{record['index']}"
        )

        print(
            f"Question: "
            f"{record['question']}"
        )

        print(
            "Baseline oracle failure: "
            f"{record['baseline_failure']}"
        )

        print(
            "Residual oracle failure: "
            f"{record['residual_failure']}"
        )

        print(
            f"Gold facts: "
            f"{sorted(record['gold'])}"
        )

        print(
            f"Missing after expansion: "
            f"{sorted(record['missing'])}"
        )

        print(
            "Missing-fact causes: "
            f"{dict(record['cause_counts'])}"
        )

        for fact in sorted(
            record["missing"]
        ):

            print(
                f"\n  Missing fact: "
                f"{fact}"
            )

            locations = (
                record[
                    "locations"
                ][fact]
            )

            if not locations:

                print(
                    "    candidate@100: "
                    "NOT AVAILABLE"
                )

                continue

            print(
                "    candidate@100: "
                "AVAILABLE"
            )

            for location in locations:

                chunk = location[
                    "chunk"
                ]

                print(
                    "    "
                    f"chunk={chunk.id} "
                    f"dense_rank="
                    f"{format_rank(location['dense_rank'])} "
                    f"bm25_rank="
                    f"{format_rank(location['sparse_rank'])} "
                    f"title="
                    f"{chunk.metadata.get('title', '<unknown>')!r} "
                    f"sentence_ids="
                    f"{chunk.metadata.get('sentence_ids', [])}"
                )

        print(
            "\n  Expanded top-10:"
        )

        for result in (
            record[
                "expanded_results"
            ]
        ):

            chunk = result.chunk

            matched = (
                get_chunk_evidence(
                    chunk
                )
                & record["gold"]
            )

            print(
                "    "
                f"rank={result.rank} "
                f"score={result.score:.4f} "
                f"dense_rank="
                f"{format_rank(result.dense_rank)} "
                f"bm25_rank="
                f"{format_rank(result.sparse_rank)} "
                f"title="
                f"{chunk.metadata.get('title', '<unknown>')!r} "
                f"gold_match="
                f"{sorted(matched)}"
            )

    # --------------------------------------------------------
    # Consistency
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "CONSISTENCY CHECKS"
    )

    print("=" * 100)

    assert baseline_failures == 63

    assert residual_failures == 53

    assert (
        baseline_failures
        - residual_failures
        == 10
    )

    assert (
        sum(
            baseline_failure_types.values()
        )
        == baseline_failures
    )

    assert (
        sum(
            residual_failure_types.values()
        )
        == residual_failures
    )

    assert total_missing_facts > 0

    assert (
        missing_fact_causes[
            "unexpected"
        ]
        == 0
    )

    print(
        "\nPASS: Baseline failures reproduce "
        "Phase 10.2B (63)."
    )

    print(
        "PASS: Expanded-candidate residuals "
        "reproduce Phase 10.2B (53)."
    )

    print(
        "PASS: Exactly 10 baseline failures "
        "were repaired."
    )

    print(
        "PASS: Every residual missing gold "
        "fact received a cause classification."
    )

    print(
        "PASS: No missing fact was unexpectedly "
        "present in final top-10."
    )

    print(
        "\nNOTE: This is offline oracle failure "
        "analysis. Gold evidence is not a runtime "
        "recovery feature."
    )


if __name__ == "__main__":
    main()