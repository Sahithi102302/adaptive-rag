from __future__ import annotations

from collections import Counter

from datasets import load_dataset

from src.chunking.sentence_chunker import (
    SentenceChunker,
    SentenceChunkingConfig,
)
from src.diagnosis.oracle import (
    diagnose_oracle_retrieval,
)
from src.diagnosis.states import (
    FailureType,
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
CANDIDATE_K = 20
FINAL_K = 10


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


def main() -> None:

    print("=" * 90)
    print(
        "PHASE 8.4 — CROSS-ENCODER "
        "ORACLE DIAGNOSIS"
    )
    print("=" * 90)

    # --------------------------------------------------------
    # Same 100-question development slice
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
    # Diagnosis counters
    # --------------------------------------------------------

    diagnosis_counts = Counter()

    total_questions = 0
    total_gold_facts = 0

    sufficient = 0
    incomplete = 0

    examples_by_type: dict[
        FailureType,
        list[dict],
    ] = {
        failure_type: []
        for failure_type in FailureType
    }

    # --------------------------------------------------------
    # Run best current retrieval pipeline
    # --------------------------------------------------------

    print(
        "\nRunning cross-encoder retrieval "
        "and oracle diagnosis...\n"
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
            top_k=FINAL_K,
        )

        diagnosis = diagnose_oracle_retrieval(
            gold_evidence=gold,
            results=results,
        )

        total_questions += 1
        total_gold_facts += len(gold)

        diagnosis_counts[
            diagnosis.failure_type
        ] += 1

        if (
            diagnosis.failure_type
            == FailureType.SUFFICIENT_EVIDENCE
        ):
            sufficient += 1
        else:
            incomplete += 1

        signals = diagnosis.signals

        examples_by_type[
            diagnosis.failure_type
        ].append(
            {
                "question": (
                    example.question
                ),
                "failure_type": (
                    diagnosis.failure_type.value
                ),
                "gold_facts": (
                    signals.gold_fact_count
                ),
                "retrieved_gold_facts": (
                    signals.retrieved_gold_fact_count
                ),
                "gold_documents": (
                    signals.gold_document_count
                ),
                "retrieved_gold_documents": (
                    signals.retrieved_gold_document_count
                ),
                "partial": (
                    signals.has_partial_gold_evidence
                ),
                "complete": (
                    signals.has_complete_gold_evidence
                ),
                "missing_document": (
                    signals.has_missing_gold_document
                ),
                "wrong_passage": (
                    signals.has_gold_document_wrong_passage
                ),
                "explanation": (
                    diagnosis.explanation
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
    # Distribution
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "CROSS-ENCODER ORACLE "
        "DIAGNOSIS DISTRIBUTION"
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

    for failure_type in FailureType:

        count = diagnosis_counts[
            failure_type
        ]

        percentage = (
            count
            / total_questions
        )

        print(
            f"{failure_type.value}: "
            f"{count} "
            f"({percentage:.1%})"
        )

    print(
        f"\nSufficient evidence: "
        f"{sufficient}"
    )

    print(
        f"Incomplete evidence: "
        f"{incomplete}"
    )

    # --------------------------------------------------------
    # Compare with original dense oracle
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "DENSE BASELINE VS "
        "CROSS-ENCODER ORACLE STATES"
    )

    print(
        "=" * 90
    )

    print(
        "\nOriginal Dense@10 "
        "oracle distribution:"
    )

    print(
        "  sufficient_evidence: "
        "64"
    )

    print(
        "  severe_retrieval_failure: "
        "1"
    )

    print(
        "  document_selection_failure: "
        "0"
    )

    print(
        "  passage_selection_failure: "
        "5"
    )

    print(
        "  evidence_coverage_failure: "
        "30"
    )

    print(
        "\nCurrent Cross-Encoder@10 "
        "oracle distribution:"
    )

    for failure_type in (
        FailureType.SUFFICIENT_EVIDENCE,
        FailureType.SEVERE_RETRIEVAL_FAILURE,
        FailureType.DOCUMENT_SELECTION_FAILURE,
        FailureType.PASSAGE_SELECTION_FAILURE,
        FailureType.EVIDENCE_COVERAGE_FAILURE,
    ):

        print(
            f"  {failure_type.value}: "
            f"{diagnosis_counts[failure_type]}"
        )

    # --------------------------------------------------------
    # Residual failure examples
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "RESIDUAL FAILURE EXAMPLES"
    )

    print(
        "=" * 90
    )

    residual_types = (
        FailureType.SEVERE_RETRIEVAL_FAILURE,
        FailureType.DOCUMENT_SELECTION_FAILURE,
        FailureType.PASSAGE_SELECTION_FAILURE,
        FailureType.EVIDENCE_COVERAGE_FAILURE,
        FailureType.UNKNOWN,
    )

    for failure_type in residual_types:

        cases = examples_by_type[
            failure_type
        ]

        print(
            f"\n--- "
            f"{failure_type.value} "
            f"({len(cases)}) ---"
        )

        if not cases:
            print(
                "No examples."
            )
            continue

        for case in cases[:3]:

            print(
                f"\nQuestion: "
                f"{case['question']}"
            )

            print(
                "Gold facts: "
                f"{case['gold_facts']}"
            )

            print(
                "Retrieved gold facts: "
                f"{case['retrieved_gold_facts']}"
            )

            print(
                "Gold documents: "
                f"{case['gold_documents']}"
            )

            print(
                "Retrieved gold documents: "
                f"{case['retrieved_gold_documents']}"
            )

            print(
                "Partial evidence: "
                f"{case['partial']}"
            )

            print(
                "Missing gold document: "
                f"{case['missing_document']}"
            )

            print(
                "Gold doc / wrong passage: "
                f"{case['wrong_passage']}"
            )

            print(
                "Explanation: "
                f"{case['explanation']}"
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

    # Phase 8.2.
    assert sufficient == 75
    assert incomplete == 25

    assert (
        sum(
            diagnosis_counts.values()
        )
        == total_questions
    )

    assert (
        sufficient
        + incomplete
        == total_questions
    )

    # These semantic states are not currently
    # produced by the Hotpot retrieval oracle.
    assert (
        diagnosis_counts[
            FailureType.AMBIGUOUS_QUERY
        ]
        == 0
    )

    assert (
        diagnosis_counts[
            FailureType.CONFLICTING_EVIDENCE
        ]
        == 0
    )

    assert (
        diagnosis_counts[
            FailureType.TEMPORAL_FAILURE
        ]
        == 0
    )

    assert (
        diagnosis_counts[
            FailureType.UNSUPPORTED_INFERENCE
        ]
        == 0
    )

    print(
        "\nPASS: Evaluated the expected "
        "100 questions."
    )

    print(
        "PASS: Evaluated the expected "
        "249 valid gold facts."
    )

    print(
        "PASS: Cross-Encoder reproduces "
        "Phase 8.2 complete/incomplete "
        "split (75/25)."
    )

    print(
        "PASS: Every question received "
        "exactly one oracle state."
    )

    print(
        "\nNOTE: These are offline oracle "
        "diagnoses because HotpotQA gold "
        "evidence is used."
    )


if __name__ == "__main__":
    main()