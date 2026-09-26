from __future__ import annotations

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
from src.retrieval.reranker import (
    CrossEncoderReranker,
    RerankerConfig,
)
from src.retrieval.sparse_retriever import (
    BM25Retriever,
)


NUM_EXAMPLES = 100
TOP_K_VALUES = (1, 5, 10)
MAX_K = max(TOP_K_VALUES)

CANDIDATE_K = 20


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


def get_chunk_evidence(
    chunk,
) -> set[tuple[str, int]]:

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


def main() -> None:

    print("=" * 90)
    print(
        "PHASE 8.2 — CROSS-ENCODER "
        "RERANKING EVALUATION"
    )
    print("=" * 90)

    # --------------------------------------------------------
    # Same development slice
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
    # Same sentence-aware chunking
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
    # Dense retrieval
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
    # Cross-encoder reranker
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

    print(
        f"Candidate depth per retriever: "
        f"{CANDIDATE_K}"
    )

    print(
        f"Final evaluation depth: "
        f"{MAX_K}"
    )

    # --------------------------------------------------------
    # Metric accumulators
    # --------------------------------------------------------

    total_questions = 0
    total_gold_facts = 0

    evidence_hits = {
        k: 0
        for k in TOP_K_VALUES
    }

    question_hits = {
        k: 0
        for k in TOP_K_VALUES
    }

    complete_evidence_hits = {
        k: 0
        for k in TOP_K_VALUES
    }

    reciprocal_rank_sum = 0.0

    incomplete_examples = []

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print(
        "\nEvaluating cross-encoder "
        "reranking...\n"
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

        results = reranker.search(
            query=example.question,
            embedder=embedder,
            top_k=MAX_K,
        )

        total_questions += 1
        total_gold_facts += len(gold)

        cumulative_evidence: set[
            tuple[str, int]
        ] = set()

        evidence_by_rank = []

        first_gold_rank = None

        for result in results:

            chunk_evidence = (
                get_chunk_evidence(
                    result.chunk
                )
            )

            matched = (
                gold
                & chunk_evidence
            )

            cumulative_evidence.update(
                matched
            )

            evidence_by_rank.append(
                set(cumulative_evidence)
            )

            if (
                matched
                and first_gold_rank is None
            ):
                first_gold_rank = (
                    result.rank
                )

        # Defensive padding.
        while (
            len(evidence_by_rank)
            < MAX_K
        ):
            evidence_by_rank.append(
                set(cumulative_evidence)
            )

        for k in TOP_K_VALUES:

            retrieved_at_k = (
                evidence_by_rank[k - 1]
            )

            evidence_hits[k] += len(
                gold
                & retrieved_at_k
            )

            if (
                gold
                & retrieved_at_k
            ):
                question_hits[k] += 1

            if gold.issubset(
                retrieved_at_k
            ):
                complete_evidence_hits[
                    k
                ] += 1

        if (
            first_gold_rank is not None
            and first_gold_rank <= MAX_K
        ):
            reciprocal_rank_sum += (
                1.0
                / first_gold_rank
            )

        final_retrieved = (
            evidence_by_rank[
                MAX_K - 1
            ]
        )

        if not gold.issubset(
            final_retrieved
        ):
            incomplete_examples.append(
                (
                    example.question,
                    gold,
                    final_retrieved,
                    results,
                )
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
    # Metrics
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "CROSS-ENCODER RERANKING METRICS"
    )

    print(
        "=" * 90
    )

    print(
        f"\nEvaluated questions: "
        f"{total_questions}"
    )

    print(
        f"Valid gold evidence facts: "
        f"{total_gold_facts}"
    )

    for k in TOP_K_VALUES:

        evidence_recall = (
            evidence_hits[k]
            / total_gold_facts
        )

        question_hit_rate = (
            question_hits[k]
            / total_questions
        )

        complete_rate = (
            complete_evidence_hits[k]
            / total_questions
        )

        print(
            f"\n@{k}"
        )

        print(
            "  Evidence Recall: "
            f"{evidence_recall:.4f} "
            f"({evidence_hits[k]}/"
            f"{total_gold_facts})"
        )

        print(
            "  Question Hit: "
            f"{question_hit_rate:.4f} "
            f"({question_hits[k]}/"
            f"{total_questions})"
        )

        print(
            "  Complete Evidence: "
            f"{complete_rate:.4f} "
            f"({complete_evidence_hits[k]}/"
            f"{total_questions})"
        )

    mrr = (
        reciprocal_rank_sum
        / total_questions
    )

    print(
        f"\nMRR@{MAX_K}: "
        f"{mrr:.4f}"
    )

    print(
        f"Incomplete evidence@{MAX_K}: "
        f"{len(incomplete_examples)}/"
        f"{total_questions}"
    )

    # --------------------------------------------------------
    # Baseline comparison
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "RETRIEVAL / RERANKING COMPARISON @10"
    )

    print(
        "=" * 90
    )

    ce_recall = (
        evidence_hits[10]
        / total_gold_facts
    )

    ce_complete = (
        complete_evidence_hits[10]
        / total_questions
    )

    print(
        "\nDense:"
    )
    print(
        "  Evidence Recall@10: "
        "0.8153 (203/249)"
    )
    print(
        "  Complete Evidence@10: "
        "0.6400 (64/100)"
    )
    print(
        "  MRR@10: 0.9000"
    )

    print(
        "\nBM25:"
    )
    print(
        "  Evidence Recall@10: "
        "0.8554 (213/249)"
    )
    print(
        "  Complete Evidence@10: "
        "0.7000 (70/100)"
    )
    print(
        "  MRR@10: 0.8707"
    )

    print(
        "\nHybrid RRF:"
    )
    print(
        "  Evidence Recall@10: "
        "0.8394 (209/249)"
    )
    print(
        "  Complete Evidence@10: "
        "0.6800 (68/100)"
    )
    print(
        "  MRR@10: 0.9126"
    )

    print(
        "\nCross-Encoder:"
    )
    print(
        "  Evidence Recall@10: "
        f"{ce_recall:.4f} "
        f"({evidence_hits[10]}/"
        f"{total_gold_facts})"
    )
    print(
        "  Complete Evidence@10: "
        f"{ce_complete:.4f} "
        f"({complete_evidence_hits[10]}/"
        f"{total_questions})"
    )
    print(
        "  MRR@10: "
        f"{mrr:.4f}"
    )

    print(
        "\nCandidate availability ceiling "
        "from Dense@20 union BM25@20:"
    )
    print(
        "  Evidence availability: "
        "0.9478 (236/249)"
    )

    print(
        "\nNOTE: Candidate availability is "
        "an offline oracle analysis ceiling, "
        "not deployable top-10 performance."
    )

    # --------------------------------------------------------
    # Sample incomplete cases
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "SAMPLE CROSS-ENCODER "
        "INCOMPLETE-EVIDENCE CASES"
    )

    print(
        "=" * 90
    )

    for case_index, (
        question,
        gold,
        retrieved,
        results,
    ) in enumerate(
        incomplete_examples[:5],
        start=1,
    ):

        missing = (
            gold
            - retrieved
        )

        print(
            f"\n--- Failure "
            f"{case_index} ---"
        )

        print(
            f"Question: "
            f"{question}"
        )

        print(
            f"Gold facts: "
            f"{len(gold)}"
        )

        print(
            f"Gold facts found: "
            f"{len(gold & retrieved)}"
        )

        print(
            "Missing gold facts: "
            f"{sorted(missing)}"
        )

        print(
            "\nTop reranked results:"
        )

        for result in results[:5]:

            title = (
                result.chunk.metadata.get(
                    "title",
                    "<unknown>",
                )
            )

            sentence_ids = (
                result.chunk.metadata.get(
                    "sentence_ids",
                    [],
                )
            )

            print(
                f"  rank={result.rank} "
                f"ce_score="
                f"{result.score:.6f} "
                f"dense_rank="
                f"{result.dense_rank} "
                f"bm25_rank="
                f"{result.sparse_rank} "
                f"title={title!r} "
                f"sentence_ids="
                f"{sentence_ids}"
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

    assert (
        evidence_hits[1]
        <= evidence_hits[5]
        <= evidence_hits[10]
    )

    assert (
        question_hits[1]
        <= question_hits[5]
        <= question_hits[10]
    )

    assert (
        complete_evidence_hits[1]
        <= complete_evidence_hits[5]
        <= complete_evidence_hits[10]
    )

    # A final reranker cannot recover gold
    # evidence absent from its candidate pool.
    assert evidence_hits[10] <= 236

    print(
        "\nPASS: Evaluated the expected "
        "100 questions."
    )

    print(
        "PASS: Evaluated the expected "
        "249 valid gold facts."
    )

    print(
        "PASS: Metrics are monotonic "
        "with retrieval depth."
    )

    print(
        "PASS: Cross-encoder top-10 evidence "
        "does not exceed candidate-pool "
        "availability."
    )

    print(
        "\nNOTE: This remains a development "
        "sanity evaluation, not the final "
        "held-out benchmark."
    )


if __name__ == "__main__":
    main()