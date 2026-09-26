from datasets import load_dataset

from src.chunking.sentence_chunker import (
    SentenceChunker,
    SentenceChunkingConfig,
    build_evidence_chunk_index,
)
from src.ingestion.hotpotqa import (
    build_hotpotqa_corpus,
)


def main() -> None:
    print("=" * 70)
    print("AdaptiveRAG - Full Sentence Chunking Validation")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load the full HotpotQA training split
    # ---------------------------------------------------------

    print("\nLoading HotpotQA...")

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    train_dataset = dataset["train"]

    print(
        f"Training examples available: "
        f"{len(train_dataset):,}"
    )

    # ---------------------------------------------------------
    # 2. Build the version-aware corpus
    # ---------------------------------------------------------

    print(
        "\nBuilding full version-aware "
        "HotpotQA corpus..."
    )

    corpus = build_hotpotqa_corpus(
        train_dataset
    )

    print(
        f"QA examples: "
        f"{len(corpus.examples):,}"
    )

    print(
        f"Unique document versions: "
        f"{len(corpus.documents):,}"
    )

    documents_by_id = {
        document.id: document
        for document in corpus.documents
    }

    # ---------------------------------------------------------
    # 3. Sentence-aware chunking
    # ---------------------------------------------------------

    config = SentenceChunkingConfig(
        sentences_per_chunk=3,
        sentence_overlap=1,
    )

    print(
        "\nSentence chunking configuration:"
    )

    print(
        f"  Sentences per chunk: "
        f"{config.sentences_per_chunk}"
    )

    print(
        f"  Sentence overlap:    "
        f"{config.sentence_overlap}"
    )

    chunker = SentenceChunker(config)

    print(
        "\nChunking full corpus..."
    )

    chunks = chunker.chunk_documents(
        corpus.documents
    )

    print(
        f"Generated chunks: "
        f"{len(chunks):,}"
    )

    # ---------------------------------------------------------
    # 4. Basic chunk structural validation
    # ---------------------------------------------------------

    print(
        "\nValidating chunk structure..."
    )

    empty_chunks = 0
    chunks_without_sentence_ids = 0
    invalid_chunk_sentence_ids = 0

    for chunk in chunks:
        if not chunk.text.strip():
            empty_chunks += 1

        sentence_ids = chunk.metadata.get(
            "sentence_ids",
            [],
        )

        if not sentence_ids:
            chunks_without_sentence_ids += 1
            continue

        document = documents_by_id.get(
            chunk.document_id
        )

        if document is None:
            invalid_chunk_sentence_ids += 1
            continue

        source_sentences = document.metadata.get(
            "sentences",
            [],
        )

        for sentence_id in sentence_ids:
            if not (
                0
                <= sentence_id
                < len(source_sentences)
            ):
                invalid_chunk_sentence_ids += 1

    # ---------------------------------------------------------
    # 5. Build evidence -> chunk lookup
    # ---------------------------------------------------------

    print(
        "\nBuilding evidence-to-chunk index..."
    )

    evidence_index = (
        build_evidence_chunk_index(
            chunks
        )
    )

    print(
        f"Evidence index entries: "
        f"{len(evidence_index):,}"
    )

    # ---------------------------------------------------------
    # 6. Validate every gold supporting fact
    # ---------------------------------------------------------

    print(
        "\nValidating all gold "
        "supporting facts..."
    )

    total_gold_facts = 0
    valid_gold_facts = 0
    mapped_valid_gold_facts = 0

    missing_documents = 0
    invalid_sentence_ids = 0
    empty_gold_sentences = 0

    unmapped_valid_gold_facts = 0

    malformed_facts_that_mapped = 0

    malformed_examples: list[dict] = []
    unmapped_examples: list[dict] = []

    for example in corpus.examples:
        for fact in example.supporting_facts:
            total_gold_facts += 1

            document = documents_by_id.get(
                fact.document_id
            )

            if document is None:
                missing_documents += 1

                key = (
                    fact.document_id,
                    fact.sentence_id,
                )

                if key in evidence_index:
                    malformed_facts_that_mapped += 1

                continue

            sentences = document.metadata.get(
                "sentences",
                [],
            )

            if not (
                0
                <= fact.sentence_id
                < len(sentences)
            ):
                invalid_sentence_ids += 1

                key = (
                    fact.document_id,
                    fact.sentence_id,
                )

                if key in evidence_index:
                    malformed_facts_that_mapped += 1

                if len(malformed_examples) < 10:
                    malformed_examples.append(
                        {
                            "example_id": example.id,
                            "title": fact.title,
                            "document_id": (
                                fact.document_id
                            ),
                            "sentence_id": (
                                fact.sentence_id
                            ),
                            "sentence_count": (
                                len(sentences)
                            ),
                        }
                    )

                continue

            gold_sentence = sentences[
                fact.sentence_id
            ]

            if not gold_sentence.strip():
                empty_gold_sentences += 1
                continue

            valid_gold_facts += 1

            key = (
                fact.document_id,
                fact.sentence_id,
            )

            matching_chunk_ids = (
                evidence_index.get(
                    key,
                    [],
                )
            )

            if matching_chunk_ids:
                mapped_valid_gold_facts += 1

            else:
                unmapped_valid_gold_facts += 1

                if len(unmapped_examples) < 10:
                    unmapped_examples.append(
                        {
                            "example_id": example.id,
                            "title": fact.title,
                            "document_id": (
                                fact.document_id
                            ),
                            "sentence_id": (
                                fact.sentence_id
                            ),
                            "gold_sentence": (
                                gold_sentence
                            ),
                        }
                    )

    # ---------------------------------------------------------
    # 7. Report
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("CHUNK STRUCTURE")
    print("=" * 70)

    print(
        f"Generated chunks:              "
        f"{len(chunks):,}"
    )

    print(
        f"Empty chunks:                  "
        f"{empty_chunks:,}"
    )

    print(
        f"Chunks without sentence IDs:   "
        f"{chunks_without_sentence_ids:,}"
    )

    print(
        f"Invalid chunk sentence IDs:    "
        f"{invalid_chunk_sentence_ids:,}"
    )

    print(
        f"Evidence index entries:        "
        f"{len(evidence_index):,}"
    )

    print("\n" + "=" * 70)
    print("GOLD EVIDENCE VALIDATION")
    print("=" * 70)

    print(
        f"Total gold facts:              "
        f"{total_gold_facts:,}"
    )

    print(
        f"Valid gold facts:              "
        f"{valid_gold_facts:,}"
    )

    print(
        f"Mapped valid gold facts:       "
        f"{mapped_valid_gold_facts:,}"
    )

    print(
        f"Missing documents:             "
        f"{missing_documents:,}"
    )

    print(
        f"Invalid sentence IDs:          "
        f"{invalid_sentence_ids:,}"
    )

    print(
        f"Empty gold sentences:          "
        f"{empty_gold_sentences:,}"
    )

    print(
        f"Unmapped valid gold facts:     "
        f"{unmapped_valid_gold_facts:,}"
    )

    print(
        f"Malformed facts that mapped:   "
        f"{malformed_facts_that_mapped:,}"
    )

    if valid_gold_facts:
        mapping_rate = (
            mapped_valid_gold_facts
            / valid_gold_facts
        ) * 100

        print(
            f"Valid evidence mapping rate:   "
            f"{mapping_rate:.4f}%"
        )

    # ---------------------------------------------------------
    # 8. Show real malformed annotations
    # ---------------------------------------------------------

    if malformed_examples:
        print("\n" + "=" * 70)
        print("SAMPLE MALFORMED GOLD FACTS")
        print("=" * 70)

        for problem in malformed_examples:
            print(
                f"\nExample ID: "
                f"{problem['example_id']}"
            )

            print(
                f"Title: "
                f"{problem['title']}"
            )

            print(
                f"Document ID: "
                f"{problem['document_id']}"
            )

            print(
                f"Gold sentence ID: "
                f"{problem['sentence_id']}"
            )

            print(
                f"Actual sentence count: "
                f"{problem['sentence_count']}"
            )

    # ---------------------------------------------------------
    # 9. Show unexpected mapping failures
    # ---------------------------------------------------------

    if unmapped_examples:
        print("\n" + "=" * 70)
        print("UNEXPECTED UNMAPPED VALID FACTS")
        print("=" * 70)

        for problem in unmapped_examples:
            print(
                f"\nExample ID: "
                f"{problem['example_id']}"
            )

            print(
                f"Title: "
                f"{problem['title']}"
            )

            print(
                f"Sentence ID: "
                f"{problem['sentence_id']}"
            )

            print(
                f"Gold sentence: "
                f"{problem['gold_sentence']}"
            )

    # ---------------------------------------------------------
    # 10. Final assertions
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    assert empty_chunks == 0, (
        "Sentence chunker produced empty chunks."
    )

    assert chunks_without_sentence_ids == 0, (
        "Some chunks do not contain sentence IDs."
    )

    assert invalid_chunk_sentence_ids == 0, (
        "Some chunks contain out-of-range sentence IDs."
    )

    assert missing_documents == 0, (
        "Gold evidence references missing documents."
    )

    assert unmapped_valid_gold_facts == 0, (
        "Some valid gold evidence was lost during chunking."
    )

    assert malformed_facts_that_mapped == 0, (
        "Malformed benchmark evidence accidentally mapped "
        "to retrieval chunks."
    )

    assert (
        mapped_valid_gold_facts
        == valid_gold_facts
    ), (
        "Not every valid gold supporting fact mapped "
        "to a chunk."
    )

    print(
        "PASS: Sentence-aware chunking preserved "
        "all valid gold evidence."
    )

    print(
        "PASS: Malformed gold evidence was not "
        "silently repaired or mapped."
    )

    print(
        "PASS: Full corpus chunk structure is valid."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()