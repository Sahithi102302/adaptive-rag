from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

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


NUM_EXAMPLES = 100
TOP_K = 10

EvidenceKey = Tuple[str, int]


@dataclass
class FailureRecord:
    question: str

    gold_facts: int
    retrieved_gold_facts: int

    gold_documents: int
    retrieved_gold_documents: int

    missing_gold_facts: Set[EvidenceKey]
    missing_gold_documents: Set[str]

    gold_document_present: bool
    wrong_passage_in_gold_document: bool

    no_gold_evidence: bool
    partial_evidence: bool


def get_gold_evidence(
    example,
) -> Set[EvidenceKey]:

    gold: Set[EvidenceKey] = set()

    for fact in example.supporting_facts:

        if fact.document_id is None:
            continue

        if fact.sentence_id < 0:
            continue

        gold.add(
            (
                fact.document_id,
                fact.sentence_id,
            )
        )

    return gold


def get_retrieved_evidence(
    results,
) -> Set[EvidenceKey]:

    retrieved: Set[EvidenceKey] = set()

    for result in results:

        chunk = result.chunk

        sentence_ids = chunk.metadata.get(
            "sentence_ids",
            [],
        )

        for sentence_id in sentence_ids:
            retrieved.add(
                (
                    chunk.document_id,
                    sentence_id,
                )
            )

    return retrieved


def get_retrieved_documents(
    results,
) -> Set[str]:

    return {
        result.chunk.document_id
        for result in results
    }


def find_wrong_passage_failure(
    gold: Set[EvidenceKey],
    retrieved: Set[EvidenceKey],
    retrieved_documents: Set[str],
) -> bool:
    """
    Detect whether at least one missing gold sentence belongs
    to a document that WAS retrieved.

    This means retrieval reached the correct document but did
    not retrieve a chunk containing the required sentence.
    """

    missing = gold - retrieved

    for document_id, _ in missing:

        if document_id in retrieved_documents:
            return True

    return False


def first_document_ranks(
    results,
    gold_documents: Set[str],
) -> Dict[str, int | None]:

    ranks: Dict[str, int | None] = {
        document_id: None
        for document_id in gold_documents
    }

    for result in results:

        document_id = (
            result.chunk.document_id
        )

        if (
            document_id in ranks
            and ranks[document_id] is None
        ):
            ranks[document_id] = result.rank

    return ranks


def main() -> None:

    print("=" * 90)
    print(
        "PHASE 6.4 — STRUCTURED DENSE-RETRIEVAL "
        "FAILURE CHARACTERIZATION"
    )
    print("=" * 90)

    # --------------------------------------------------------
    # 1. Build the SAME development corpus as Phase 6.3
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
    # 2. Same sentence-aware chunking
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

    # --------------------------------------------------------
    # 3. Same retrieval representation
    # --------------------------------------------------------

    retrieval_texts = [
        format_chunk_for_retrieval(
            chunk
        )
        for chunk in chunks
    ]

    # --------------------------------------------------------
    # 4. Same embedding model
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
        f"Embedding dimension: "
        f"{embedder.embedding_dimension}"
    )

    print(
        f"\nEmbedding "
        f"{len(retrieval_texts):,} chunks..."
    )

    chunk_embeddings = (
        embedder.encode_documents(
            retrieval_texts,
            show_progress_bar=True,
        )
    )

    # --------------------------------------------------------
    # 5. Same exact FAISS baseline
    # --------------------------------------------------------

    print(
        "\nBuilding exact FAISS index..."
    )

    retriever = DenseRetriever(
        chunks=chunks,
        embeddings=chunk_embeddings,
    )

    print(
        f"Indexed chunks: "
        f"{retriever.size:,}"
    )

    # --------------------------------------------------------
    # 6. Characterize incomplete-evidence questions
    # --------------------------------------------------------

    failures: List[FailureRecord] = []

    failure_type_counts = Counter()

    print(
        "\nAnalyzing retrieval failures...\n"
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

        results = retriever.search(
            query=example.question,
            embedder=embedder,
            top_k=TOP_K,
        )

        retrieved = get_retrieved_evidence(
            results
        )

        matched = (
            gold
            & retrieved
        )

        # Complete retrieval -> not a failure
        if gold.issubset(retrieved):
            continue

        gold_documents = {
            document_id
            for document_id, _
            in gold
        }

        retrieved_documents = (
            get_retrieved_documents(
                results
            )
        )

        retrieved_gold_documents = (
            gold_documents
            & retrieved_documents
        )

        missing_gold_documents = (
            gold_documents
            - retrieved_documents
        )

        missing_gold_facts = (
            gold
            - retrieved
        )

        wrong_passage = (
            find_wrong_passage_failure(
                gold=gold,
                retrieved=retrieved,
                retrieved_documents=(
                    retrieved_documents
                ),
            )
        )

        no_gold_evidence = (
            len(matched) == 0
        )

        partial_evidence = (
            0
            < len(matched)
            < len(gold)
        )

        gold_document_present = (
            len(retrieved_gold_documents)
            > 0
        )

        # ----------------------------------------------------
        # Observable failure signals
        # ----------------------------------------------------

        if no_gold_evidence:
            failure_type_counts[
                "no_gold_evidence"
            ] += 1

        if partial_evidence:
            failure_type_counts[
                "partial_evidence"
            ] += 1

        if missing_gold_documents:
            failure_type_counts[
                "missing_gold_document"
            ] += 1

        if wrong_passage:
            failure_type_counts[
                "correct_document_wrong_passage"
            ] += 1

        failures.append(
            FailureRecord(
                question=example.question,

                gold_facts=len(gold),

                retrieved_gold_facts=(
                    len(matched)
                ),

                gold_documents=(
                    len(gold_documents)
                ),

                retrieved_gold_documents=(
                    len(
                        retrieved_gold_documents
                    )
                ),

                missing_gold_facts=(
                    missing_gold_facts
                ),

                missing_gold_documents=(
                    missing_gold_documents
                ),

                gold_document_present=(
                    gold_document_present
                ),

                wrong_passage_in_gold_document=(
                    wrong_passage
                ),

                no_gold_evidence=(
                    no_gold_evidence
                ),

                partial_evidence=(
                    partial_evidence
                ),
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

    # ========================================================
    # 7. Summary
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "FAILURE SUMMARY"
    )

    print(
        "=" * 90
    )

    print(
        f"\nIncomplete-evidence questions: "
        f"{len(failures)}"
    )

    print(
        "\nObservable failure signals:"
    )

    for name, count in (
        failure_type_counts.most_common()
    ):
        percentage = (
            count
            / len(failures)
            * 100
            if failures
            else 0.0
        )

        print(
            f"  {name}: "
            f"{count} "
            f"({percentage:.1f}%)"
        )

    # ========================================================
    # 8. Failure structure
    # ========================================================

    all_gold_docs_found = 0
    some_gold_docs_found = 0
    no_gold_docs_found = 0

    for failure in failures:

        if (
            failure.retrieved_gold_documents
            == failure.gold_documents
        ):
            all_gold_docs_found += 1

        elif (
            failure.retrieved_gold_documents
            > 0
        ):
            some_gold_docs_found += 1

        else:
            no_gold_docs_found += 1

    print(
        "\nGold-document retrieval status:"
    )

    print(
        f"  All gold documents retrieved: "
        f"{all_gold_docs_found}"
    )

    print(
        f"  Some gold documents retrieved: "
        f"{some_gold_docs_found}"
    )

    print(
        f"  No gold documents retrieved: "
        f"{no_gold_docs_found}"
    )

    # ========================================================
    # 9. Detailed sample failures
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "DETAILED FAILURE EXAMPLES"
    )

    print(
        "=" * 90
    )

    for failure_number, failure in enumerate(
        failures[:10],
        start=1,
    ):

        print(
            f"\n--- Failure "
            f"{failure_number} ---"
        )

        print(
            f"Question: "
            f"{failure.question}"
        )

        print(
            f"Gold evidence facts: "
            f"{failure.gold_facts}"
        )

        print(
            f"Retrieved gold facts: "
            f"{failure.retrieved_gold_facts}"
        )

        print(
            f"Gold documents: "
            f"{failure.gold_documents}"
        )

        print(
            f"Retrieved gold documents: "
            f"{failure.retrieved_gold_documents}"
        )

        print(
            f"Missing gold documents: "
            f"{sorted(failure.missing_gold_documents)}"
        )

        print(
            f"Missing gold facts: "
            f"{sorted(failure.missing_gold_facts)}"
        )

        print(
            f"No gold evidence: "
            f"{failure.no_gold_evidence}"
        )

        print(
            f"Partial evidence: "
            f"{failure.partial_evidence}"
        )

        print(
            "Correct document but wrong passage: "
            f"{failure.wrong_passage_in_gold_document}"
        )

    # ========================================================
    # 10. Interpretation guardrail
    # ========================================================

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
        "\nThese are observable retrieval signals, "
        "not yet final semantic failure labels."
    )

    print(
        "Signals can overlap. For example, a question "
        "may have one missing gold document AND another "
        "gold document retrieved with the wrong passage."
    )

    print(
        "\nThe next step is to use these signals to "
        "design mutually meaningful diagnostic states "
        "and map those states to candidate recovery actions."
    )


if __name__ == "__main__":
    main()