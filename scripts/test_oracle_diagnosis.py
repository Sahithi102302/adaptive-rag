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


def main() -> None:

    print("=" * 90)
    print(
        "PHASE 6.5B — ORACLE DIAGNOSIS VALIDATION"
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

    retriever = DenseRetriever(
        chunks=chunks,
        embeddings=chunk_embeddings,
    )

    counts = Counter()

    examples_by_type = {}

    print(
        "\nRunning oracle diagnosis...\n"
    )

    evaluated = 0

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
            embedder=embedder,
            top_k=TOP_K,
        )

        diagnosis = (
            diagnose_oracle_retrieval(
                gold_evidence=gold,
                results=results,
            )
        )

        counts[
            diagnosis.failure_type
        ] += 1

        evaluated += 1

        examples_by_type.setdefault(
            diagnosis.failure_type,
            (
                example.question,
                diagnosis,
            ),
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

    print(
        "\n"
        + "=" * 90
    )

    print(
        "ORACLE DIAGNOSIS DISTRIBUTION"
    )

    print(
        "=" * 90
    )

    print(
        f"\nEvaluated questions: "
        f"{evaluated}"
    )

    for failure_type in FailureType:

        count = counts[
            failure_type
        ]

        percentage = (
            count
            / evaluated
            * 100
            if evaluated
            else 0.0
        )

        print(
            f"{failure_type.value}: "
            f"{count} "
            f"({percentage:.1f}%)"
        )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "ONE EXAMPLE PER OBSERVED STATE"
    )

    print(
        "=" * 90
    )

    for (
        failure_type,
        (
            question,
            diagnosis,
        ),
    ) in examples_by_type.items():

        print(
            f"\nState: "
            f"{failure_type.value}"
        )

        print(
            f"Question: "
            f"{question}"
        )

        print(
            f"Family: "
            f"{diagnosis.failure_family.value}"
        )

        print(
            f"Recovery recommended: "
            f"{diagnosis.recovery_recommended}"
        )

        print(
            f"Explanation: "
            f"{diagnosis.explanation}"
        )

        print(
            "Signals:"
        )

        print(
            f"  gold facts: "
            f"{diagnosis.signals.gold_fact_count}"
        )

        print(
            f"  retrieved gold facts: "
            f"{diagnosis.signals.retrieved_gold_fact_count}"
        )

        print(
            f"  gold documents: "
            f"{diagnosis.signals.gold_document_count}"
        )

        print(
            f"  retrieved gold documents: "
            f"{diagnosis.signals.retrieved_gold_document_count}"
        )

        print(
            f"  partial evidence: "
            f"{diagnosis.signals.has_partial_gold_evidence}"
        )

        print(
            f"  missing gold document: "
            f"{diagnosis.signals.has_missing_gold_document}"
        )

        print(
            "  correct document / wrong passage: "
            f"{diagnosis.signals.has_gold_document_wrong_passage}"
        )

    # --------------------------------------------------------
    # Structural checks against Phase 6.3 / 6.4
    # --------------------------------------------------------

    sufficient = counts[
        FailureType.SUFFICIENT_EVIDENCE
    ]

    incomplete = (
        evaluated
        - sufficient
    )

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

    print(
        f"\nSufficient evidence: "
        f"{sufficient}"
    )

    print(
        f"Incomplete evidence: "
        f"{incomplete}"
    )

    assert evaluated == 100

    assert sufficient == 64

    assert incomplete == 36

    print(
        "\nPASS: Oracle diagnosis reproduces "
        "the Phase 6.3 complete/incomplete split."
    )

    print(
        "PASS: Every evaluated question receives "
        "exactly one oracle diagnostic state."
    )


if __name__ == "__main__":
    main()