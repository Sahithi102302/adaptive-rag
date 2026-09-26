from __future__ import annotations

from datasets import load_dataset

from src.chunking.sentence_chunker import (
    SentenceChunker,
    SentenceChunkingConfig,
)
from src.ingestion.hotpotqa import (
    build_hotpotqa_corpus,
)
from src.retrieval.dense_retriever import (
    format_chunk_for_retrieval,
)
from src.retrieval.sparse_retriever import (
    BM25Retriever,
)


NUM_EXAMPLES = 100
TOP_K_VALUES = (1, 5, 10)
MAX_K = max(TOP_K_VALUES)


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
        "PHASE 7.2 — BM25 SPARSE "
        "RETRIEVAL EVALUATION"
    )
    print("=" * 90)

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
    # Sentence-aware chunking
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
    # Build BM25
    # --------------------------------------------------------

    print(
        "\nBuilding BM25 index..."
    )

    retriever = BM25Retriever(
        chunks=chunks,
        retrieval_texts=retrieval_texts,
    )

    print(
        f"Indexed chunks: "
        f"{retriever.size:,}"
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

    print(
        "\nEvaluating BM25 retrieval...\n"
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    for index, example in enumerate(
        corpus.examples,
        start=1,
    ):

        gold = get_gold_evidence(
            example
        )

        if not gold:
            continue

        results = retriever.search(
            query=example.question,
            top_k=MAX_K,
        )

        total_questions += 1
        total_gold_facts += len(gold)

        cumulative_evidence: set[
            tuple[str, int]
        ] = set()

        first_gold_rank = None

        evidence_by_rank = []

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

        # If fewer than MAX_K results somehow
        # exist, pad with final cumulative state.
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
    # Results
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "BM25 RETRIEVAL METRICS"
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
    # Small failure sample
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "SAMPLE INCOMPLETE-EVIDENCE CASES"
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
            f"Missing gold facts: "
            f"{sorted(missing)}"
        )

        print(
            "\nTop results:"
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
                f"score={result.score:.4f} "
                f"title={title!r} "
                f"sentence_ids="
                f"{sentence_ids}"
            )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "INTERPRETATION NOTE"
    )

    print(
        "=" * 90
    )

    print(
        "\nThis is a 100-question development "
        "sanity evaluation."
    )

    print(
        "It is not the final held-out benchmark "
        "and should not be used for tuning."
    )


if __name__ == "__main__":
    main()