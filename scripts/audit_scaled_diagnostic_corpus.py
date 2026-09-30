from __future__ import annotations

from collections import Counter

from datasets import (
    concatenate_datasets,
    load_dataset,
)

from src.chunking.sentence_chunker import (
    SentenceChunker,
    SentenceChunkingConfig,
)
from src.ingestion.hotpotqa import (
    build_hotpotqa_corpus,
)


# =============================================================
# Experiment design
# =============================================================

EXPLORATORY_START = 0
EXPLORATORY_END = 100

TRAIN_START = 100
TRAIN_END = 1100

VALIDATION_START = 1100
VALIDATION_END = 1300

TEST_START = 0
TEST_END = 500


def valid_gold_evidence(
    example,
) -> set[tuple[str, int]]:
    """
    Return valid HotpotQA supporting-fact keys.

    Used only for corpus auditing. These annotations will later
    remain oracle labels and must never enter runtime predictor
    features.
    """

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


def document_ids(
    corpus,
) -> set[str]:
    """
    Return document IDs contained in a HotpotQA corpus.
    """

    return {
        document.id
        for document in corpus.documents
    }


def print_split_summary(
    name: str,
    corpus,
) -> None:
    """
    Print structural statistics for one query split.
    """

    gold_counts = [
        len(
            valid_gold_evidence(
                example
            )
        )
        for example in corpus.examples
    ]

    empty_gold = sum(
        count == 0
        for count in gold_counts
    )

    total_gold = sum(
        gold_counts
    )

    print(f"\n{name}")
    print("-" * 70)

    print(
        f"Questions: "
        f"{len(corpus.examples):,}"
    )

    print(
        f"Unique documents: "
        f"{len(corpus.documents):,}"
    )

    print(
        f"Valid gold facts: "
        f"{total_gold:,}"
    )

    print(
        f"Questions with zero valid "
        f"gold facts: {empty_gold:,}"
    )


def main() -> None:

    print("=" * 90)
    print(
        "PHASE 9.5A — SCALED DIAGNOSTIC "
        "CORPUS AUDIT"
    )
    print("=" * 90)

    # ---------------------------------------------------------
    # Load HotpotQA
    # ---------------------------------------------------------

    print("\nLoading HotpotQA...")

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]

    print(
        f"Full train examples: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Full validation examples: "
        f"{len(validation_dataset):,}"
    )

    # ---------------------------------------------------------
    # Construct query subsets
    #
    # build_hotpotqa_corpus currently accepts max_examples,
    # so for non-zero starts we use dataset.select(...) first.
    # ---------------------------------------------------------

    exploratory_raw = train_dataset.select(
        range(
            EXPLORATORY_START,
            EXPLORATORY_END,
        )
    )

    detector_train_raw = train_dataset.select(
        range(
            TRAIN_START,
            TRAIN_END,
        )
    )

    detector_validation_raw = (
        train_dataset.select(
            range(
                VALIDATION_START,
                VALIDATION_END,
            )
        )
    )

    detector_test_raw = (
        validation_dataset.select(
            range(
                TEST_START,
                TEST_END,
            )
        )
    )

    # ---------------------------------------------------------
    # Build each corpus independently for auditing
    # ---------------------------------------------------------

    print(
        "\nBuilding exploratory corpus..."
    )

    exploratory_corpus = (
        build_hotpotqa_corpus(
            exploratory_raw,
        )
    )

    print(
        "Building detector-training corpus..."
    )

    detector_train_corpus = (
        build_hotpotqa_corpus(
            detector_train_raw,
        )
    )

    print(
        "Building detector-validation corpus..."
    )

    detector_validation_corpus = (
        build_hotpotqa_corpus(
            detector_validation_raw,
        )
    )

    print(
        "Building detector-test corpus..."
    )

    detector_test_corpus = (
        build_hotpotqa_corpus(
            detector_test_raw,
        )
    )

    # ---------------------------------------------------------
    # Split summaries
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print("QUERY SPLIT SUMMARY")
    print("=" * 90)

    print_split_summary(
        "Exploratory development",
        exploratory_corpus,
    )

    print_split_summary(
        "Detector train",
        detector_train_corpus,
    )

    print_split_summary(
        "Detector validation",
        detector_validation_corpus,
    )

    print_split_summary(
        "Detector test",
        detector_test_corpus,
    )

    # ---------------------------------------------------------
    # Query-count assertions
    # ---------------------------------------------------------

    assert (
        len(exploratory_corpus.examples)
        == 100
    )

    assert (
        len(detector_train_corpus.examples)
        == 1000
    )

    assert (
        len(detector_validation_corpus.examples)
        == 200
    )

    assert (
        len(detector_test_corpus.examples)
        == 500
    )

    # ---------------------------------------------------------
    # Query-source separation
    #
    # These are positional ranges:
    #
    # exploratory       train[0:100]
    # detector train    train[100:1100]
    # detector val      train[1100:1300]
    # detector test     validation[0:500]
    #
    # Therefore the train-derived query ranges are disjoint.
    # Test queries come from HotpotQA validation.
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print("QUERY SOURCE DESIGN")
    print("=" * 90)

    print(
        "Exploratory: "
        "HotpotQA train[0:100]"
    )

    print(
        "Detector train: "
        "HotpotQA train[100:1100]"
    )

    print(
        "Detector validation: "
        "HotpotQA train[1100:1300]"
    )

    print(
        "Detector test: "
        "HotpotQA validation[0:500]"
    )

    # ---------------------------------------------------------
    # Document overlap
    #
    # Document overlap is expected and is not label leakage.
    # HotpotQA contexts can recur across different questions.
    # We measure it explicitly so the benchmark construction is
    # transparent.
    # ---------------------------------------------------------

    exploratory_docs = document_ids(
        exploratory_corpus
    )

    train_docs = document_ids(
        detector_train_corpus
    )

    validation_docs = document_ids(
        detector_validation_corpus
    )

    test_docs = document_ids(
        detector_test_corpus
    )

    print("\n" + "=" * 90)
    print("DOCUMENT OVERLAP")
    print("=" * 90)

    overlap_pairs = [
        (
            "exploratory / train",
            exploratory_docs,
            train_docs,
        ),
        (
            "exploratory / validation",
            exploratory_docs,
            validation_docs,
        ),
        (
            "exploratory / test",
            exploratory_docs,
            test_docs,
        ),
        (
            "train / validation",
            train_docs,
            validation_docs,
        ),
        (
            "train / test",
            train_docs,
            test_docs,
        ),
        (
            "validation / test",
            validation_docs,
            test_docs,
        ),
    ]

    for (
        name,
        left,
        right,
    ) in overlap_pairs:

        overlap = left & right

        print(
            f"{name}: "
            f"{len(overlap):,}"
        )

    # ---------------------------------------------------------
    # Build one pooled benchmark-derived corpus
    #
    # We combine the selected examples BEFORE corpus building.
    # This allows build_hotpotqa_corpus() to perform its normal
    # version-aware document deduplication globally.
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print("POOLED CORPUS")
    print("=" * 90)

    pooled_raw = concatenate_datasets(
    [
        exploratory_raw,
        detector_train_raw,
        detector_validation_raw,
        detector_test_raw,
    ]
)

    print(
        f"Pooled query examples: "
        f"{len(pooled_raw):,}"
    )

    pooled_corpus = (
        build_hotpotqa_corpus(
            pooled_raw
        )
    )

    print(
        f"Pooled corpus examples: "
        f"{len(pooled_corpus.examples):,}"
    )

    print(
        f"Pooled unique documents: "
        f"{len(pooled_corpus.documents):,}"
    )

    # ---------------------------------------------------------
    # Sentence-aware chunking
    # ---------------------------------------------------------

    print(
        "\nSentence-aware chunking "
        "pooled corpus..."
    )

    chunker = SentenceChunker(
        SentenceChunkingConfig(
            sentences_per_chunk=3,
            sentence_overlap=1,
        )
    )

    chunks = chunker.chunk_documents(
        pooled_corpus.documents
    )

    print(
        f"Pooled sentence chunks: "
        f"{len(chunks):,}"
    )

    # ---------------------------------------------------------
    # Document multiplicity across query splits
    #
    # Number of pooled documents appearing in exactly N of the
    # four query-source corpora.
    # ---------------------------------------------------------

    split_document_sets = [
        exploratory_docs,
        train_docs,
        validation_docs,
        test_docs,
    ]

    pooled_document_ids = document_ids(
        pooled_corpus
    )

    multiplicity = Counter()

    for document_id in pooled_document_ids:

        appearances = sum(
            document_id in split_docs
            for split_docs
            in split_document_sets
        )

        multiplicity[
            appearances
        ] += 1

    print("\nDocument split multiplicity:")

    for appearances in sorted(
        multiplicity
    ):

        print(
            f"  Appears in "
            f"{appearances} split(s): "
            f"{multiplicity[appearances]:,}"
        )

    # ---------------------------------------------------------
    # Approximate dense embedding memory
    #
    # MiniLM dimension = 384
    # float32 = 4 bytes
    # ---------------------------------------------------------

    embedding_dimension = 384
    bytes_per_float = 4

    embedding_bytes = (
        len(chunks)
        * embedding_dimension
        * bytes_per_float
    )

    embedding_mib = (
        embedding_bytes
        / (1024 ** 2)
    )

    embedding_gib = (
        embedding_bytes
        / (1024 ** 3)
    )

    print("\nApproximate embedding matrix size:")

    print(
        f"  {embedding_mib:,.2f} MiB"
    )

    print(
        f"  {embedding_gib:,.3f} GiB"
    )

    # ---------------------------------------------------------
    # Final checks
    # ---------------------------------------------------------

    assert (
        len(pooled_corpus.examples)
        == 1800
    )

    assert len(chunks) > 0

    print("\n" + "=" * 90)

    print(
        "PASS: Scaled diagnostic corpus "
        "audit completed successfully."
    )

    print(
        "NOTE: No embedding model, FAISS index, "
        "BM25 index, or cross-encoder was loaded."
    )


if __name__ == "__main__":
    main()