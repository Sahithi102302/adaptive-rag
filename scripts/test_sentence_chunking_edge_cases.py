from src.chunking.sentence_chunker import (
    SentenceChunker,
    SentenceChunkingConfig,
    find_chunks_for_evidence,
)
from src.ingestion.document import Document


def test_empty_sentence_preservation() -> None:
    """
    Verify that empty sentence positions remain represented by their
    original sentence IDs even though empty text is not included in
    the chunk text.
    """

    print("=" * 70)
    print("TEST 1 - EMPTY SENTENCE POSITION PRESERVATION")
    print("=" * 70)

    document = Document(
        id="edge_case_empty_sentence",
        text="Sentence zero. Sentence two. Sentence three.",
        metadata={
            "title": "Empty Sentence Test",
            "sentences": [
                "Sentence zero.",
                "",
                "Sentence two.",
                "Sentence three.",
            ],
        },
    )

    chunker = SentenceChunker(
        SentenceChunkingConfig(
            sentences_per_chunk=3,
            sentence_overlap=1,
        )
    )

    chunks = chunker.chunk_document(document)

    print("\nOriginal sentence positions:")

    for sentence_id, sentence in enumerate(
        document.metadata["sentences"]
    ):
        print(f"[{sentence_id}] {sentence!r}")

    print("\nGenerated chunks:")

    for chunk in chunks:
        print("-" * 70)
        print(f"Chunk ID: {chunk.id}")
        print(
            f"Sentence IDs: "
            f"{chunk.metadata['sentence_ids']}"
        )
        print(f"Text: {chunk.text}")

    assert len(chunks) == 2

    assert chunks[0].metadata["sentence_ids"] == [
        0,
        1,
        2,
    ]

    assert chunks[1].metadata["sentence_ids"] == [
        2,
        3,
    ]

    assert chunks[0].text == (
        "Sentence zero. Sentence two."
    )

    assert 1 in chunks[0].metadata["sentence_ids"]

    print(
        "\nPASS: Empty sentence position 1 was preserved "
        "without inserting empty text into the chunk."
    )


def test_fully_empty_window() -> None:
    """
    Verify that a sentence window containing no usable text does not
    create an empty retrieval chunk.
    """

    print("\n" + "=" * 70)
    print("TEST 2 - FULLY EMPTY WINDOW")
    print("=" * 70)

    document = Document(
        id="edge_case_empty_window",
        text="Placeholder",
        metadata={
            "title": "Fully Empty Window Test",
            "sentences": [
                "",
                "",
                "",
            ],
        },
    )

    chunker = SentenceChunker(
        SentenceChunkingConfig(
            sentences_per_chunk=3,
            sentence_overlap=1,
        )
    )

    chunks = chunker.chunk_document(document)

    print(f"\nGenerated chunks: {len(chunks)}")

    assert len(chunks) == 0

    print(
        "PASS: A fully empty sentence window did not "
        "produce an empty retrieval chunk."
    )


def test_overlap_mapping() -> None:
    """
    Verify that an overlapping sentence maps to every chunk that
    contains its sentence position.
    """

    print("\n" + "=" * 70)
    print("TEST 3 - OVERLAPPING EVIDENCE MAPPING")
    print("=" * 70)

    document = Document(
        id="edge_case_overlap",
        text=(
            "Sentence zero. Sentence one. Sentence two. "
            "Sentence three. Sentence four."
        ),
        metadata={
            "title": "Overlap Test",
            "sentences": [
                "Sentence zero.",
                "Sentence one.",
                "Sentence two.",
                "Sentence three.",
                "Sentence four.",
            ],
        },
    )

    chunker = SentenceChunker(
        SentenceChunkingConfig(
            sentences_per_chunk=3,
            sentence_overlap=1,
        )
    )

    chunks = chunker.chunk_document(document)

    for chunk in chunks:
        print(
            f"{chunk.id}: "
            f"{chunk.metadata['sentence_ids']}"
        )

    matches = find_chunks_for_evidence(
        chunks=chunks,
        document_id=document.id,
        sentence_id=2,
    )

    print("\nChunks containing sentence 2:")

    for chunk in matches:
        print(
            f"  {chunk.id}: "
            f"{chunk.metadata['sentence_ids']}"
        )

    assert len(matches) == 2

    assert matches[0].metadata["sentence_ids"] == [
        0,
        1,
        2,
    ]

    assert matches[1].metadata["sentence_ids"] == [
        2,
        3,
        4,
    ]

    print(
        "\nPASS: Overlapping evidence maps to every "
        "chunk containing that sentence."
    )


def test_invalid_sentence_id() -> None:
    """
    Verify that a malformed/out-of-range evidence sentence ID does
    not accidentally map to a chunk.
    """

    print("\n" + "=" * 70)
    print("TEST 4 - INVALID SENTENCE ID")
    print("=" * 70)

    document = Document(
        id="edge_case_invalid_id",
        text="Sentence zero. Sentence one.",
        metadata={
            "title": "Invalid Sentence ID Test",
            "sentences": [
                "Sentence zero.",
                "Sentence one.",
            ],
        },
    )

    chunker = SentenceChunker(
        SentenceChunkingConfig(
            sentences_per_chunk=3,
            sentence_overlap=1,
        )
    )

    chunks = chunker.chunk_document(document)

    invalid_sentence_id = 20

    matches = find_chunks_for_evidence(
        chunks=chunks,
        document_id=document.id,
        sentence_id=invalid_sentence_id,
    )

    print(
        f"\nAttempted sentence ID: "
        f"{invalid_sentence_id}"
    )
    print(f"Matching chunks: {len(matches)}")

    assert len(matches) == 0

    print(
        "PASS: Invalid sentence ID did not map "
        "to any chunk."
    )


def test_wrong_document_id() -> None:
    """
    Verify that sentence identity alone is insufficient: evidence
    must also belong to the correct document.
    """

    print("\n" + "=" * 70)
    print("TEST 5 - DOCUMENT ID ISOLATION")
    print("=" * 70)

    document = Document(
        id="correct_document",
        text="Sentence zero. Sentence one.",
        metadata={
            "title": "Document Isolation Test",
            "sentences": [
                "Sentence zero.",
                "Sentence one.",
            ],
        },
    )

    chunker = SentenceChunker(
        SentenceChunkingConfig(
            sentences_per_chunk=3,
            sentence_overlap=1,
        )
    )

    chunks = chunker.chunk_document(document)

    matches = find_chunks_for_evidence(
        chunks=chunks,
        document_id="wrong_document",
        sentence_id=0,
    )

    print(f"\nMatching chunks: {len(matches)}")

    assert len(matches) == 0

    print(
        "PASS: Evidence did not map across "
        "different document IDs."
    )


def main() -> None:
    print("\n")
    print("#" * 70)
    print("AdaptiveRAG - Sentence Chunking Edge-Case Validation")
    print("#" * 70)
    print()

    test_empty_sentence_preservation()
    test_fully_empty_window()
    test_overlap_mapping()
    test_invalid_sentence_id()
    test_wrong_document_id()

    print("\n" + "#" * 70)
    print("ALL EDGE-CASE TESTS PASSED")
    print("#" * 70)


if __name__ == "__main__":
    main()