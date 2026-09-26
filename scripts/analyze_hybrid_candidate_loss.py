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
        "PHASE 7.5 — HYBRID "
        "CANDIDATE-LOSS ANALYSIS"
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
    # Same sentence chunking
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
    # Sparse
    # --------------------------------------------------------

    print(
        "\nBuilding BM25 index..."
    )

    sparse_retriever = BM25Retriever(
        chunks=chunks,
        retrieval_texts=retrieval_texts,
    )

    # --------------------------------------------------------
    # Hybrid
    # --------------------------------------------------------

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        config=HybridRetrievalConfig(
            candidate_k=CANDIDATE_K,
            rrf_k=RRF_K,
        ),
    )

    print(
        f"Candidate depth: "
        f"{CANDIDATE_K}"
    )

    print(
        f"Final hybrid depth: "
        f"{FINAL_K}"
    )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total_questions = 0
    total_gold_facts = 0

    hybrid_complete = 0
    hybrid_incomplete = 0

    failure_types = Counter()

    total_candidate_gold = 0
    total_final_gold = 0

    lost_gold_facts = 0
    unavailable_gold_facts = 0

    ranking_loss_cases = []
    candidate_failure_cases = []
    mixed_failure_cases = []

    # --------------------------------------------------------
    # Analysis
    # --------------------------------------------------------

    print(
        "\nAnalyzing candidate loss...\n"
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

        # Retrieve the exact candidate pools used
        # by the hybrid system.
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

        hybrid_results = (
            hybrid_retriever.search(
                query=example.question,
                embedder=embedder,
                top_k=FINAL_K,
            )
        )

        dense_gold = get_retrieved_gold(
            dense_candidates,
            gold,
        )

        sparse_gold = get_retrieved_gold(
            sparse_candidates,
            gold,
        )

        candidate_gold = (
            dense_gold
            | sparse_gold
        )

        final_gold = get_retrieved_gold(
            hybrid_results,
            gold,
        )

        total_questions += 1
        total_gold_facts += len(gold)

        total_candidate_gold += len(
            candidate_gold
        )

        total_final_gold += len(
            final_gold
        )

        if gold.issubset(
            final_gold
        ):
            hybrid_complete += 1
            continue

        hybrid_incomplete += 1

        # Gold facts that entered at least one
        # top-20 candidate list but disappeared
        # from the final hybrid top-10.
        lost_after_fusion = (
            candidate_gold
            - final_gold
        )

        # Gold facts absent from both top-20
        # candidate lists.
        unavailable = (
            gold
            - candidate_gold
        )

        lost_gold_facts += len(
            lost_after_fusion
        )

        unavailable_gold_facts += len(
            unavailable
        )

        if (
            lost_after_fusion
            and not unavailable
        ):
            failure_type = (
                "ranking_loss_only"
            )

        elif (
            unavailable
            and not lost_after_fusion
        ):
            failure_type = (
                "candidate_failure_only"
            )

        elif (
            lost_after_fusion
            and unavailable
        ):
            failure_type = (
                "mixed"
            )

        else:
            # Defensive category. An incomplete
            # question should normally have at
            # least one missing fact explained by
            # one of the categories above.
            failure_type = (
                "unclassified"
            )

        failure_types[
            failure_type
        ] += 1

        case = {
            "question": (
                example.question
            ),
            "gold_count": len(
                gold
            ),
            "candidate_found": len(
                candidate_gold
            ),
            "final_found": len(
                final_gold
            ),
            "dense_candidate_found": len(
                dense_gold
            ),
            "bm25_candidate_found": len(
                sparse_gold
            ),
            "lost_after_fusion": sorted(
                lost_after_fusion
            ),
            "unavailable": sorted(
                unavailable
            ),
        }

        if (
            failure_type
            == "ranking_loss_only"
        ):
            ranking_loss_cases.append(
                case
            )

        elif (
            failure_type
            == "candidate_failure_only"
        ):
            candidate_failure_cases.append(
                case
            )

        elif failure_type == "mixed":
            mixed_failure_cases.append(
                case
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
    # Overall candidate availability
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "CANDIDATE-POOL EVIDENCE AVAILABILITY"
    )

    print(
        "=" * 90
    )

    print(
        f"\nTotal gold facts: "
        f"{total_gold_facts}"
    )

    print(
        "Gold facts available in "
        "Dense@20 ∪ BM25@20: "
        f"{total_candidate_gold}/"
        f"{total_gold_facts} "
        f"({total_candidate_gold / total_gold_facts:.4f})"
    )

    print(
        "Gold facts retained in "
        "Hybrid RRF@10: "
        f"{total_final_gold}/"
        f"{total_gold_facts} "
        f"({total_final_gold / total_gold_facts:.4f})"
    )

    print(
        "Gold facts lost during "
        "candidate -> final ranking: "
        f"{lost_gold_facts}"
    )

    print(
        "Gold facts unavailable from "
        "both candidate lists: "
        f"{unavailable_gold_facts}"
    )

    # --------------------------------------------------------
    # Question-level failures
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "HYBRID FAILURE ATTRIBUTION"
    )

    print(
        "=" * 90
    )

    print(
        f"\nHybrid complete@10: "
        f"{hybrid_complete}/"
        f"{total_questions}"
    )

    print(
        f"Hybrid incomplete@10: "
        f"{hybrid_incomplete}/"
        f"{total_questions}"
    )

    for key, label in (
        (
            "ranking_loss_only",
            "Ranking/fusion loss only",
        ),
        (
            "candidate_failure_only",
            "Candidate retrieval failure only",
        ),
        (
            "mixed",
            "Mixed candidate + ranking failure",
        ),
        (
            "unclassified",
            "Unclassified",
        ),
    ):

        count = failure_types[key]

        denominator = max(
            hybrid_incomplete,
            1,
        )

        print(
            f"{label}: "
            f"{count} "
            f"({count / denominator:.1%} "
            "of incomplete questions)"
        )

    # --------------------------------------------------------
    # Examples
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

            print(
                "Question: "
                f"{case['question']}"
            )

            print(
                "Gold facts: "
                f"{case['gold_count']}"
            )

            print(
                "Dense@20 gold found: "
                f"{case['dense_candidate_found']}"
            )

            print(
                "BM25@20 gold found: "
                f"{case['bm25_candidate_found']}"
            )

            print(
                "Union candidate gold found: "
                f"{case['candidate_found']}"
            )

            print(
                "Hybrid@10 gold found: "
                f"{case['final_found']}"
            )

            print(
                "Lost after fusion: "
                f"{case['lost_after_fusion']}"
            )

            print(
                "Unavailable from candidates: "
                f"{case['unavailable']}"
            )

    print_cases(
        "RANKING / FUSION LOSS EXAMPLES",
        ranking_loss_cases,
    )

    print_cases(
        "CANDIDATE RETRIEVAL FAILURE EXAMPLES",
        candidate_failure_cases,
    )

    print_cases(
        "MIXED FAILURE EXAMPLES",
        mixed_failure_cases,
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

    # Reproduce Phase 7.4B.
    assert hybrid_complete == 68
    assert hybrid_incomplete == 32
    assert total_final_gold == 209

    assert (
        hybrid_complete
        + hybrid_incomplete
        == total_questions
    )

    assert (
        sum(
            failure_types.values()
        )
        == hybrid_incomplete
    )

    assert (
        total_candidate_gold
        >= total_final_gold
    )

    print(
        "\nPASS: Hybrid RRF reproduces "
        "Phase 7.4B."
    )

    print(
        "PASS: Every incomplete question "
        "received one failure category."
    )

    print(
        "PASS: Candidate evidence availability "
        "is not lower than final Hybrid@10."
    )

    print(
        "\nNOTE: This is an offline oracle "
        "analysis because gold evidence is used "
        "to attribute failures."
    )


if __name__ == "__main__":
    main()