from datasets import load_dataset

from src.chunking.sentence_chunker import (
    SentenceChunker,
    SentenceChunkingConfig,
    find_chunks_for_evidence,
)
from src.ingestion.hotpotqa import build_hotpotqa_corpus


def main() -> None:
    print("=" * 70)
    print("AdaptiveRAG - Evidence-Aware Sentence Chunking Test")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load HotpotQA
    # ---------------------------------------------------------

    print("\nLoading HotpotQA...")

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    train_dataset = dataset["train"]

    print(
        f"Available training examples: "
        f"{len(train_dataset):,}"
    )

    # ---------------------------------------------------------
    # 2. Build a small development corpus
    # ---------------------------------------------------------

    print("\nBuilding small HotpotQA corpus...")

    corpus = build_hotpotqa_corpus(
        train_dataset,
        max_examples=100,
    )

    print(f"QA examples: {len(corpus.examples):,}")
    print(
        f"Unique document versions: "
        f"{len(corpus.documents):,}"
    )

    # corpus.documents is a list[Document].
    # Build a lookup dictionary for efficient document-ID access
    # during gold-evidence validation.
    documents_by_id = {
        document.id: document
        for document in corpus.documents
    }

    # ---------------------------------------------------------
    # 3. Configure sentence-aware chunking
    # ---------------------------------------------------------

    config = SentenceChunkingConfig(
        sentences_per_chunk=3,
        sentence_overlap=1,
    )

    chunker = SentenceChunker(config)

    print("\nSentence chunking configuration:")
    print(
        f"  Sentences per chunk: "
        f"{config.sentences_per_chunk}"
    )
    print(
        f"  Sentence overlap:    "
        f"{config.sentence_overlap}"
    )

    # ---------------------------------------------------------
    # 4. Chunk all documents
    # ---------------------------------------------------------

    print("\nChunking documents...")

    chunks = chunker.chunk_documents(
        corpus.documents
    )

    print(f"Generated chunks: {len(chunks):,}")

    # ---------------------------------------------------------
    # 5. Inspect one source document
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAMPLE DOCUMENT")
    print("=" * 70)

    first_document = corpus.documents[0]

    print(f"Document ID: {first_document.id}")
    print(
        f"Title: "
        f"{first_document.metadata.get('title')}"
    )

    sentences = first_document.metadata.get(
        "sentences",
        [],
    )

    print(f"Sentence count: {len(sentences)}")

    print("\nOriginal sentences:")

    for sentence_id, sentence in enumerate(sentences):
        print(
            f"[{sentence_id}] "
            f"{sentence!r}"
        )

    # ---------------------------------------------------------
    # 6. Inspect chunks generated from that document
    # ---------------------------------------------------------

    document_chunks = [
        chunk
        for chunk in chunks
        if chunk.document_id == first_document.id
    ]

    print("\nGenerated sentence-aware chunks:")

    for chunk in document_chunks:
        print("-" * 70)

        print(f"Chunk ID: {chunk.id}")

        print(
            "Sentence IDs:",
            chunk.metadata["sentence_ids"],
        )

        print(
            "Sentence range:",
            f"{chunk.metadata['sentence_start']}"
            f" -> "
            f"{chunk.metadata['sentence_end']}",
        )

        print(
            f"Character count: "
            f"{chunk.character_count}"
        )

        print(f"Text: {chunk.text}")

    # ---------------------------------------------------------
    # 7. Validate gold evidence -> chunk mapping
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("GOLD EVIDENCE MAPPING TEST")
    print("=" * 70)

    total_gold_facts = 0
    valid_gold_facts = 0
    mapped_gold_facts = 0

    missing_documents = 0
    invalid_sentence_ids = 0
    empty_gold_sentences = 0
    unmapped_valid_facts = 0

    mapping_examples_shown = 0

    for example in corpus.examples:
        for fact in example.supporting_facts:
            total_gold_facts += 1

            # -------------------------------------------------
            # Resolve the exact document version
            # -------------------------------------------------

            document = documents_by_id.get(
                fact.document_id
            )

            if document is None:
                missing_documents += 1
                continue

            document_sentences = (
                document.metadata.get(
                    "sentences",
                    [],
                )
            )

            # -------------------------------------------------
            # Validate the sentence ID
            # -------------------------------------------------

            if not (
                0
                <= fact.sentence_id
                < len(document_sentences)
            ):
                invalid_sentence_ids += 1
                continue

            # -------------------------------------------------
            # Detect gold annotations pointing to empty text
            # -------------------------------------------------

            gold_sentence = document_sentences[
                fact.sentence_id
            ]

            if not gold_sentence.strip():
                empty_gold_sentences += 1
                continue

            valid_gold_facts += 1

            # -------------------------------------------------
            # Find all chunks containing this gold sentence
            # -------------------------------------------------

            matching_chunks = find_chunks_for_evidence(
                chunks=chunks,
                document_id=fact.document_id,
                sentence_id=fact.sentence_id,
            )

            if matching_chunks:
                mapped_gold_facts += 1
            else:
                unmapped_valid_facts += 1

            # -------------------------------------------------
            # Show a few examples for manual inspection
            # -------------------------------------------------

            if (
                matching_chunks
                and mapping_examples_shown < 5
            ):
                mapping_examples_shown += 1

                print("\nGold fact:")

                print(
                    f"  Question ID: "
                    f"{example.id}"
                )

                print(
                    f"  Title: "
                    f"{fact.title}"
                )

                print(
                    f"  Document ID: "
                    f"{fact.document_id}"
                )

                print(
                    f"  Sentence ID: "
                    f"{fact.sentence_id}"
                )

                print(
                    f"  Gold sentence: "
                    f"{gold_sentence}"
                )

                print("  Matching chunks:")

                for chunk in matching_chunks:
                    print(
                        f"    {chunk.id}"
                    )

                    print(
                        "      Sentence IDs:",
                        chunk.metadata[
                            "sentence_ids"
                        ],
                    )

    # ---------------------------------------------------------
    # 8. Mapping summary
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("MAPPING SUMMARY")
    print("=" * 70)

    print(
        f"Total gold facts:       "
        f"{total_gold_facts:,}"
    )

    print(
        f"Valid gold facts:       "
        f"{valid_gold_facts:,}"
    )

    print(
        f"Mapped gold facts:      "
        f"{mapped_gold_facts:,}"
    )

    print(
        f"Missing documents:      "
        f"{missing_documents:,}"
    )

    print(
        f"Invalid sentence IDs:   "
        f"{invalid_sentence_ids:,}"
    )

    print(
        f"Empty gold sentences:   "
        f"{empty_gold_sentences:,}"
    )

    print(
        f"Unmapped valid facts:   "
        f"{unmapped_valid_facts:,}"
    )

    # ---------------------------------------------------------
    # 9. Mapping rate
    # ---------------------------------------------------------

    if valid_gold_facts:
        mapping_rate = (
            mapped_gold_facts
            / valid_gold_facts
        ) * 100

        print(
            f"Valid evidence mapping rate: "
            f"{mapping_rate:.2f}%"
        )

    # ---------------------------------------------------------
    # 10. Final validation result
    # ---------------------------------------------------------

    print("\n" + "=" * 70)

    if (
        mapped_gold_facts == valid_gold_facts
        and unmapped_valid_facts == 0
    ):
        print(
            "PASS: Every valid gold supporting fact "
            "maps to at least one sentence-aware chunk."
        )
    else:
        print(
            "FAIL: Some valid gold supporting facts "
            "do not map to a sentence-aware chunk."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()