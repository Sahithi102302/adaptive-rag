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
from src.retrieval.sparse_retriever import (
    BM25Retriever,
)


NUM_EXAMPLES = 100
TOP_K = 10


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


def get_retrieved_gold_evidence(
    results,
    gold_evidence: set[tuple[str, int]],
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

            if key in gold_evidence:
                retrieved.add(key)

    return retrieved


def main() -> None:

    print("=" * 90)
    print(
        "PHASE 7.3 — DENSE / BM25 "
        "COMPLEMENTARITY ANALYSIS"
    )
    print("=" * 90)

    # --------------------------------------------------------
    # Load exactly the same development slice
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
    # Same sentence-aware chunks
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
    # Dense retriever
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

    chunk_embeddings = (
        embedder.encode_documents(
            retrieval_texts,
            show_progress_bar=True,
        )
    )

    dense_retriever = DenseRetriever(
        chunks=chunks,
        embeddings=chunk_embeddings,
    )

    print(
        f"Dense indexed chunks: "
        f"{len(chunks):,}"
    )

    # --------------------------------------------------------
    # BM25 retriever
    # --------------------------------------------------------

    print(
        "\nBuilding BM25 index..."
    )

    sparse_retriever = BM25Retriever(
        chunks=chunks,
        retrieval_texts=retrieval_texts,
    )

    print(
        f"BM25 indexed chunks: "
        f"{sparse_retriever.size:,}"
    )

    # --------------------------------------------------------
    # Accumulators
    # --------------------------------------------------------

    total_questions = 0
    total_gold_facts = 0

    fact_counts = Counter()

    question_counts = Counter()

    interesting_cases = []

    # --------------------------------------------------------
    # Compare retrievers
    # --------------------------------------------------------

    print(
        "\nComparing dense and BM25 retrieval...\n"
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

        dense_results = (
            dense_retriever.search(
                query=example.question,
                embedder=embedder,
                top_k=TOP_K,
            )
        )

        sparse_results = (
            sparse_retriever.search(
                query=example.question,
                top_k=TOP_K,
            )
        )

        dense_gold = (
            get_retrieved_gold_evidence(
                dense_results,
                gold,
            )
        )

        sparse_gold = (
            get_retrieved_gold_evidence(
                sparse_results,
                gold,
            )
        )

        union_gold = (
            dense_gold
            | sparse_gold
        )

        total_questions += 1
        total_gold_facts += len(gold)

        # ----------------------------------------------------
        # Evidence-fact-level complementarity
        # ----------------------------------------------------

        for fact in gold:

            in_dense = fact in dense_gold
            in_sparse = fact in sparse_gold

            if in_dense and in_sparse:
                fact_counts["both"] += 1

            elif in_dense:
                fact_counts[
                    "dense_only"
                ] += 1

            elif in_sparse:
                fact_counts[
                    "bm25_only"
                ] += 1

            else:
                fact_counts[
                    "neither"
                ] += 1

        # ----------------------------------------------------
        # Question-level complete evidence
        # ----------------------------------------------------

        dense_complete = (
            gold.issubset(
                dense_gold
            )
        )

        sparse_complete = (
            gold.issubset(
                sparse_gold
            )
        )

        union_complete = (
            gold.issubset(
                union_gold
            )
        )

        if (
            dense_complete
            and sparse_complete
        ):
            question_counts[
                "complete_both"
            ] += 1

        elif dense_complete:
            question_counts[
                "complete_dense_only"
            ] += 1

        elif sparse_complete:
            question_counts[
                "complete_bm25_only"
            ] += 1

        else:
            question_counts[
                "complete_neither"
            ] += 1

        if union_complete:
            question_counts[
                "complete_union"
            ] += 1

        # ----------------------------------------------------
        # Store cases where union adds something useful
        # ----------------------------------------------------

        dense_missing = (
            gold - dense_gold
        )

        sparse_missing = (
            gold - sparse_gold
        )

        if (
            union_complete
            and (
                not dense_complete
                or not sparse_complete
            )
        ):
            interesting_cases.append(
                {
                    "question": (
                        example.question
                    ),
                    "gold_count": len(
                        gold
                    ),
                    "dense_found": len(
                        dense_gold
                    ),
                    "bm25_found": len(
                        sparse_gold
                    ),
                    "union_found": len(
                        union_gold
                    ),
                    "dense_missing": sorted(
                        dense_missing
                    ),
                    "bm25_missing": sorted(
                        sparse_missing
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
    # Evidence-level report
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "EVIDENCE-FACT COMPLEMENTARITY @10"
    )

    print(
        "=" * 90
    )

    print(
        f"\nTotal gold evidence facts: "
        f"{total_gold_facts}"
    )

    for key, label in (
        (
            "both",
            "Retrieved by both",
        ),
        (
            "dense_only",
            "Dense only",
        ),
        (
            "bm25_only",
            "BM25 only",
        ),
        (
            "neither",
            "Retrieved by neither",
        ),
    ):

        count = fact_counts[key]

        percentage = (
            count
            / total_gold_facts
            * 100
        )

        print(
            f"{label}: "
            f"{count} "
            f"({percentage:.1f}%)"
        )

    dense_fact_total = (
        fact_counts["both"]
        + fact_counts["dense_only"]
    )

    sparse_fact_total = (
        fact_counts["both"]
        + fact_counts["bm25_only"]
    )

    union_fact_total = (
        total_gold_facts
        - fact_counts["neither"]
    )

    print(
        "\nEvidence Recall@10:"
    )

    print(
        "  Dense: "
        f"{dense_fact_total}/"
        f"{total_gold_facts} "
        f"({dense_fact_total / total_gold_facts:.4f})"
    )

    print(
        "  BM25: "
        f"{sparse_fact_total}/"
        f"{total_gold_facts} "
        f"({sparse_fact_total / total_gold_facts:.4f})"
    )

    print(
        "  Oracle union: "
        f"{union_fact_total}/"
        f"{total_gold_facts} "
        f"({union_fact_total / total_gold_facts:.4f})"
    )

    # --------------------------------------------------------
    # Question-level report
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "QUESTION-LEVEL COMPLETE EVIDENCE @10"
    )

    print(
        "=" * 90
    )

    for key, label in (
        (
            "complete_both",
            "Complete by both",
        ),
        (
            "complete_dense_only",
            "Complete only by dense",
        ),
        (
            "complete_bm25_only",
            "Complete only by BM25",
        ),
        (
            "complete_neither",
            "Complete by neither individually",
        ),
    ):

        count = question_counts[
            key
        ]

        percentage = (
            count
            / total_questions
            * 100
        )

        print(
            f"\n{label}: "
            f"{count} "
            f"({percentage:.1f}%)"
        )

    dense_complete_total = (
        question_counts[
            "complete_both"
        ]
        + question_counts[
            "complete_dense_only"
        ]
    )

    sparse_complete_total = (
        question_counts[
            "complete_both"
        ]
        + question_counts[
            "complete_bm25_only"
        ]
    )

    union_complete_total = (
        question_counts[
            "complete_union"
        ]
    )

    print(
        "\nComplete Evidence@10:"
    )

    print(
        "  Dense: "
        f"{dense_complete_total}/"
        f"{total_questions} "
        f"({dense_complete_total / total_questions:.4f})"
    )

    print(
        "  BM25: "
        f"{sparse_complete_total}/"
        f"{total_questions} "
        f"({sparse_complete_total / total_questions:.4f})"
    )

    print(
        "  Oracle union: "
        f"{union_complete_total}/"
        f"{total_questions} "
        f"({union_complete_total / total_questions:.4f})"
    )

    # --------------------------------------------------------
    # Complementary cases
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "SAMPLE COMPLEMENTARY CASES"
    )

    print(
        "=" * 90
    )

    if not interesting_cases:

        print(
            "\nNo complementary union-complete "
            "cases were found."
        )

    else:

        for case_index, case in enumerate(
            interesting_cases[:10],
            start=1,
        ):

            print(
                f"\n--- Case "
                f"{case_index} ---"
            )

            print(
                "Question: "
                f"{case['question']}"
            )

            print(
                "Gold facts: "
                f"{case['gold_count']}"
            )

            print(
                "Dense found: "
                f"{case['dense_found']}"
            )

            print(
                "BM25 found: "
                f"{case['bm25_found']}"
            )

            print(
                "Union found: "
                f"{case['union_found']}"
            )

            print(
                "Dense missing: "
                f"{case['dense_missing']}"
            )

            print(
                "BM25 missing: "
                f"{case['bm25_missing']}"
            )

    # --------------------------------------------------------
    # Sanity checks against previous phases
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

    assert dense_fact_total == 203
    assert sparse_fact_total == 213

    assert dense_complete_total == 64
    assert sparse_complete_total == 70

    assert (
        sum(
            fact_counts.values()
        )
        == total_gold_facts
    )

    assert (
        question_counts[
            "complete_both"
        ]
        + question_counts[
            "complete_dense_only"
        ]
        + question_counts[
            "complete_bm25_only"
        ]
        + question_counts[
            "complete_neither"
        ]
        == total_questions
    )

    assert (
        union_fact_total
        >= max(
            dense_fact_total,
            sparse_fact_total,
        )
    )

    assert (
        union_complete_total
        >= max(
            dense_complete_total,
            sparse_complete_total,
        )
    )

    print(
        "\nPASS: Dense metrics reproduce "
        "Phase 6.3."
    )

    print(
        "PASS: BM25 metrics reproduce "
        "Phase 7.2."
    )

    print(
        "PASS: Evidence categories form "
        "a complete partition."
    )

    print(
        "PASS: Question categories form "
        "a complete partition."
    )

    print(
        "\nNOTE: Oracle union is an analysis "
        "ceiling, not a deployable retrieval "
        "system."
    )


if __name__ == "__main__":
    main()