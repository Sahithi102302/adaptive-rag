from __future__ import annotations

from collections import Counter

from datasets import load_dataset

from src.chunking.sentence_chunker import (
    SentenceChunker,
    SentenceChunkingConfig,
)
from src.embeddings.dense_embedder import (
    DenseEmbedder,
    DenseEmbeddingConfig,
)
from src.ingestion.hotpotqa import (
    build_hotpotqa_corpus,
)
from src.retrieval.dense_retriever import (
    DenseRetriever,
    format_chunk_for_retrieval,
)
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    HybridRetrievalConfig,
)
from src.retrieval.reranker import (
    CrossEncoderReranker,
    RerankerConfig,
)
from src.retrieval.sparse_retriever import (
    BM25Retriever,
)


NUM_EXAMPLES = 100
CANDIDATE_K = 20
FINAL_K = 10
RRF_K = 60


def get_gold_evidence(
    example,
) -> set[tuple[str, int]]:

    return {
        (
            fact.document_id,
            fact.sentence_id,
        )
        for fact in example.supporting_facts
        if (
            fact.document_id is not None
            and fact.sentence_id >= 0
        )
    }


def get_retrieved_gold(
    results,
    gold: set[tuple[str, int]],
) -> set[tuple[str, int]]:

    retrieved: set[
        tuple[str, int]
    ] = set()

    for result in results:

        chunk = result.chunk

        for sentence_id in chunk.metadata.get(
            "sentence_ids",
            [],
        ):

            key = (
                chunk.document_id,
                sentence_id,
            )

            if key in gold:
                retrieved.add(key)

    return retrieved


def main() -> None:

    print("=" * 90)
    print(
        "PHASE 8.3 — RERANKER "
        "REPAIR ANALYSIS"
    )
    print("=" * 90)

    # --------------------------------------------------------
    # Same development corpus
    # --------------------------------------------------------

    print("\nLoading HotpotQA...")

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    corpus = build_hotpotqa_corpus(
        dataset["train"],
        max_examples=NUM_EXAMPLES,
    )

    print(
        f"QA examples: "
        f"{len(corpus.examples):,}"
    )

    print(
        f"Unique documents: "
        f"{len(corpus.documents):,}"
    )

    # --------------------------------------------------------
    # Same sentence chunks
    # --------------------------------------------------------

    print(
        "\nSentence-aware chunking..."
    )

    chunker = SentenceChunker(
        SentenceChunkingConfig(
            sentences_per_chunk=3,
            sentence_overlap=1,
        )
    )

    chunks = chunker.chunk_documents(
        corpus.documents
    )

    print(
        f"Chunks: "
        f"{len(chunks):,}"
    )

    retrieval_texts = [
        format_chunk_for_retrieval(
            chunk
        )
        for chunk in chunks
    ]

    # --------------------------------------------------------
    # Dense
    # --------------------------------------------------------

    print(
        "\nLoading dense embedder..."
    )

    embedder = DenseEmbedder(
        DenseEmbeddingConfig(
            model_name=(
                "sentence-transformers/"
                "all-MiniLM-L6-v2"
            ),
            batch_size=32,
            normalize_embeddings=True,
        )
    )

    print(
        f"Embedding "
        f"{len(retrieval_texts):,} chunks..."
    )

    embeddings = (
        embedder.encode_documents(
            retrieval_texts,
            show_progress_bar=True,
        )
    )

    dense_retriever = DenseRetriever(
        chunks=chunks,
        embeddings=embeddings,
    )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    print(
        "\nBuilding BM25 index..."
    )

    sparse_retriever = BM25Retriever(
        chunks=chunks,
        retrieval_texts=retrieval_texts,
    )

    # --------------------------------------------------------
    # RRF
    # --------------------------------------------------------

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        config=HybridRetrievalConfig(
            candidate_k=CANDIDATE_K,
            rrf_k=RRF_K,
        ),
    )

    # --------------------------------------------------------
    # Cross-encoder
    # --------------------------------------------------------

    print(
        "\nLoading cross-encoder..."
    )

    reranker = CrossEncoderReranker(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        config=RerankerConfig(
            model_name=(
                "cross-encoder/"
                "ms-marco-MiniLM-L-6-v2"
            ),
            candidate_k=CANDIDATE_K,
            batch_size=32,
        ),
    )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    transition_counts = Counter()

    total_questions = 0
    total_gold_facts = 0

    rrf_gold_total = 0
    ce_gold_total = 0
    candidate_gold_total = 0

    rrf_missing_fact_total = 0

    rrf_missing_available = 0
    rrf_missing_unavailable = 0

    recovered_rrf_missing_facts = 0
    still_missing_available_facts = 0

    # Question-level Phase 7.5 ranking-loss subset.
    ranking_loss_only_questions = 0
    ranking_loss_repaired = 0
    ranking_loss_still_incomplete = 0

    repair_cases = []
    persistent_ranking_cases = []
    regression_cases = []

    named_cases = []

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    print(
        "\nComparing RRF@10 and "
        "Cross-Encoder@10...\n"
    )

    for index, example in enumerate(
        corpus.examples,
        start=1,
    ):

        gold = get_gold_evidence(
            example
        )

        if not gold:
            continue

        # Actual candidate pool available to both
        # final-ranking strategies.
        dense_candidates = (
            dense_retriever.search(
                query=example.question,
                embedder=embedder,
                top_k=CANDIDATE_K,
            )
        )

        sparse_candidates = (
            sparse_retriever.search(
                query=example.question,
                top_k=CANDIDATE_K,
            )
        )

        candidate_gold = (
            get_retrieved_gold(
                dense_candidates,
                gold,
            )
            | get_retrieved_gold(
                sparse_candidates,
                gold,
            )
        )

        rrf_results = (
            hybrid_retriever.search(
                query=example.question,
                embedder=embedder,
                top_k=FINAL_K,
            )
        )

        ce_results = reranker.search(
            query=example.question,
            embedder=embedder,
            top_k=FINAL_K,
        )

        rrf_gold = get_retrieved_gold(
            rrf_results,
            gold,
        )

        ce_gold = get_retrieved_gold(
            ce_results,
            gold,
        )

        total_questions += 1
        total_gold_facts += len(gold)

        candidate_gold_total += len(
            candidate_gold
        )

        rrf_gold_total += len(
            rrf_gold
        )

        ce_gold_total += len(
            ce_gold
        )

        # ----------------------------------------------------
        # Complete/incomplete transitions
        # ----------------------------------------------------

        rrf_complete = gold.issubset(
            rrf_gold
        )

        ce_complete = gold.issubset(
            ce_gold
        )

        if rrf_complete and ce_complete:
            transition = (
                "complete_to_complete"
            )

        elif (
            not rrf_complete
            and ce_complete
        ):
            transition = (
                "incomplete_to_complete"
            )

        elif (
            rrf_complete
            and not ce_complete
        ):
            transition = (
                "complete_to_incomplete"
            )

        else:
            transition = (
                "incomplete_to_incomplete"
            )

        transition_counts[
            transition
        ] += 1

        # ----------------------------------------------------
        # Fact-level RRF loss
        # ----------------------------------------------------

        rrf_missing = (
            gold
            - rrf_gold
        )

        rrf_missing_fact_total += len(
            rrf_missing
        )

        # Missing from final RRF but present
        # somewhere in candidate pool.
        available_rrf_missing = (
            rrf_missing
            & candidate_gold
        )

        # Missing from RRF because candidate
        # generation never found it.
        unavailable_rrf_missing = (
            rrf_missing
            - candidate_gold
        )

        rrf_missing_available += len(
            available_rrf_missing
        )

        rrf_missing_unavailable += len(
            unavailable_rrf_missing
        )

        # Of the candidate-available facts RRF
        # lost, how many did CE recover?
        recovered_available = (
            available_rrf_missing
            & ce_gold
        )

        still_missing_available = (
            available_rrf_missing
            - ce_gold
        )

        recovered_rrf_missing_facts += len(
            recovered_available
        )

        still_missing_available_facts += len(
            still_missing_available
        )

        # ----------------------------------------------------
        # Reproduce Phase 7.5's pure
        # ranking/fusion-loss condition.
        #
        # RRF incomplete AND all required gold
        # evidence exists in candidate pool.
        # ----------------------------------------------------

        ranking_loss_only = (
            not rrf_complete
            and gold.issubset(
                candidate_gold
            )
        )

        if ranking_loss_only:

            ranking_loss_only_questions += 1

            case = {
                "question": (
                    example.question
                ),
                "gold_count": len(
                    gold
                ),
                "rrf_found": len(
                    rrf_gold
                ),
                "ce_found": len(
                    ce_gold
                ),
                "candidate_found": len(
                    candidate_gold
                ),
                "rrf_missing": sorted(
                    rrf_missing
                ),
                "recovered_by_ce": sorted(
                    recovered_available
                ),
                "still_missing": sorted(
                    gold - ce_gold
                ),
            }

            if ce_complete:

                ranking_loss_repaired += 1

                repair_cases.append(
                    case
                )

            else:

                ranking_loss_still_incomplete += 1

                persistent_ranking_cases.append(
                    case
                )

        # ----------------------------------------------------
        # Regression cases
        # ----------------------------------------------------

        if (
            rrf_complete
            and not ce_complete
        ):

            regression_cases.append(
                {
                    "question": (
                        example.question
                    ),
                    "gold_count": len(
                        gold
                    ),
                    "rrf_found": len(
                        rrf_gold
                    ),
                    "ce_found": len(
                        ce_gold
                    ),
                    "candidate_found": len(
                        candidate_gold
                    ),
                    "ce_missing": sorted(
                        gold - ce_gold
                    ),
                }
            )

        # ----------------------------------------------------
        # Track known diagnostic examples
        # ----------------------------------------------------

        question_lower = (
            example.question.lower()
        )

        known_case = (
            "allie goertz"
            in question_lower
            or "cadmium chloride"
            in question_lower
            or "badr hari"
            in question_lower
            or "bathurst"
            in question_lower
            or "malcolm smith"
            in question_lower
        )

        if known_case:

            named_cases.append(
                {
                    "question": (
                        example.question
                    ),
                    "gold_count": len(
                        gold
                    ),
                    "candidate_found": len(
                        candidate_gold
                    ),
                    "rrf_found": len(
                        rrf_gold
                    ),
                    "ce_found": len(
                        ce_gold
                    ),
                    "rrf_complete": (
                        rrf_complete
                    ),
                    "ce_complete": (
                        ce_complete
                    ),
                    "rrf_missing": sorted(
                        gold - rrf_gold
                    ),
                    "ce_missing": sorted(
                        gold - ce_gold
                    ),
                }
            )

        if (
            index % 10 == 0
            or index
            == len(corpus.examples)
        ):
            print(
                f"Processed "
                f"{index}/"
                f"{len(corpus.examples)}"
            )

    # --------------------------------------------------------
    # Transition matrix
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "QUESTION-LEVEL TRANSITIONS"
    )

    print(
        "=" * 90
    )

    print(
        "\nRRF complete -> "
        "CE complete: "
        f"{transition_counts['complete_to_complete']}"
    )

    print(
        "RRF incomplete -> "
        "CE complete: "
        f"{transition_counts['incomplete_to_complete']}"
    )

    print(
        "RRF complete -> "
        "CE incomplete: "
        f"{transition_counts['complete_to_incomplete']}"
    )

    print(
        "RRF incomplete -> "
        "CE incomplete: "
        f"{transition_counts['incomplete_to_incomplete']}"
    )

    print(
        "\nRRF complete total: "
        f"{transition_counts['complete_to_complete'] + transition_counts['complete_to_incomplete']}"
    )

    print(
        "Cross-Encoder complete total: "
        f"{transition_counts['complete_to_complete'] + transition_counts['incomplete_to_complete']}"
    )

    # --------------------------------------------------------
    # Fact-level repair
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "FACT-LEVEL REPAIR ANALYSIS"
    )

    print(
        "=" * 90
    )

    print(
        f"\nTotal gold facts: "
        f"{total_gold_facts}"
    )

    print(
        "Candidate-pool gold facts: "
        f"{candidate_gold_total}"
    )

    print(
        "RRF@10 gold facts: "
        f"{rrf_gold_total}"
    )

    print(
        "Cross-Encoder@10 gold facts: "
        f"{ce_gold_total}"
    )

    print(
        "\nGold facts missing from RRF@10: "
        f"{rrf_missing_fact_total}"
    )

    print(
        "RRF-missing facts available "
        "in candidate pool: "
        f"{rrf_missing_available}"
    )

    print(
        "RRF-missing facts unavailable "
        "from candidate pool: "
        f"{rrf_missing_unavailable}"
    )

    print(
        "\nCandidate-available RRF-missing "
        "facts recovered by Cross-Encoder: "
        f"{recovered_rrf_missing_facts}"
    )

    print(
        "Candidate-available RRF-missing "
        "facts still missing after "
        "Cross-Encoder: "
        f"{still_missing_available_facts}"
    )

    # --------------------------------------------------------
    # Pure ranking-loss subset
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "PHASE 7.5 RANKING-LOSS SUBSET"
    )

    print(
        "=" * 90
    )

    print(
        "\nPure RRF ranking-loss "
        "questions: "
        f"{ranking_loss_only_questions}"
    )

    print(
        "Repaired to complete by "
        "Cross-Encoder: "
        f"{ranking_loss_repaired}"
    )

    print(
        "Still incomplete after "
        "Cross-Encoder: "
        f"{ranking_loss_still_incomplete}"
    )

    if ranking_loss_only_questions:

        repair_rate = (
            ranking_loss_repaired
            / ranking_loss_only_questions
        )

        print(
            "Question-level repair rate: "
            f"{repair_rate:.4f} "
            f"({repair_rate:.1%})"
        )

    # --------------------------------------------------------
    # Examples helper
    # --------------------------------------------------------

    def print_cases(
        title: str,
        cases: list[dict],
        limit: int = 5,
    ) -> None:

        print(
            "\n"
            + "=" * 90
        )

        print(title)

        print(
            "=" * 90
        )

        if not cases:
            print(
                "\nNo cases in this category."
            )
            return

        for case_index, case in enumerate(
            cases[:limit],
            start=1,
        ):

            print(
                f"\n--- Case "
                f"{case_index} ---"
            )

            for key, value in (
                case.items()
            ):
                print(
                    f"{key}: {value}"
                )

    print_cases(
        "RRF FAILURES REPAIRED BY CROSS-ENCODER",
        repair_cases,
    )

    print_cases(
        "PERSISTENT RANKING-LOSS FAILURES",
        persistent_ranking_cases,
    )

    print_cases(
        "CROSS-ENCODER REGRESSIONS",
        regression_cases,
    )

    print_cases(
        "TRACKED DIAGNOSTIC EXAMPLES",
        named_cases,
        limit=10,
    )

    # --------------------------------------------------------
    # Consistency
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "CONSISTENCY CHECK"
    )

    print(
        "=" * 90
    )

    assert total_questions == 100
    assert total_gold_facts == 249

    # Phase 7.5.
    assert candidate_gold_total == 236

    # Phase 7.4B.
    assert rrf_gold_total == 209

    # Phase 8.2.
    assert ce_gold_total == 222

    rrf_complete_total = (
        transition_counts[
            "complete_to_complete"
        ]
        + transition_counts[
            "complete_to_incomplete"
        ]
    )

    ce_complete_total = (
        transition_counts[
            "complete_to_complete"
        ]
        + transition_counts[
            "incomplete_to_complete"
        ]
    )

    assert rrf_complete_total == 68
    assert ce_complete_total == 75

    # Phase 7.5 pure ranking/fusion loss.
    assert (
        ranking_loss_only_questions
        == 21
    )

    assert (
        ranking_loss_repaired
        + ranking_loss_still_incomplete
        == ranking_loss_only_questions
    )

    assert (
        rrf_missing_available
        + rrf_missing_unavailable
        == rrf_missing_fact_total
    )

    assert (
        recovered_rrf_missing_facts
        + still_missing_available_facts
        == rrf_missing_available
    )

    print(
        "\nPASS: Candidate pool reproduces "
        "Phase 7.5 (236/249)."
    )

    print(
        "PASS: RRF reproduces "
        "Phase 7.4B (209/249, "
        "68 complete)."
    )

    print(
        "PASS: Cross-Encoder reproduces "
        "Phase 8.2 (222/249, "
        "75 complete)."
    )

    print(
        "PASS: Pure RRF ranking-loss "
        "subset reproduces Phase 7.5 "
        "(21 questions)."
    )

    print(
        "PASS: Fact-level loss accounting "
        "is internally consistent."
    )

    print(
        "\nNOTE: Gold evidence is used here, "
        "so repair attribution remains "
        "offline oracle analysis."
    )


if __name__ == "__main__":
    main()