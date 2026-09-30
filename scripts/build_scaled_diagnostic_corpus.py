from __future__ import annotations

import json
from pathlib import Path

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


EXPLORATORY_START = 0
EXPLORATORY_END = 100

TRAIN_START = 100
TRAIN_END = 1100

VALIDATION_START = 1100
VALIDATION_END = 1300

TEST_START = 0
TEST_END = 500


OUTPUT_DIR = Path(
    "artifacts/diagnosis/scaled"
)

CHUNKS_PATH = (
    OUTPUT_DIR
    / "pooled_chunks.jsonl"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "corpus_manifest.json"
)


def main() -> None:

    print("=" * 90)
    print(
        "PHASE 9.5B — BUILD FIXED "
        "SCALED DIAGNOSTIC CORPUS"
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

    # ---------------------------------------------------------
    # Freeze exact query ranges
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

    pooled_raw = concatenate_datasets(
        [
            exploratory_raw,
            detector_train_raw,
            detector_validation_raw,
            detector_test_raw,
        ]
    )

    print(
        f"Pooled questions: "
        f"{len(pooled_raw):,}"
    )

    # ---------------------------------------------------------
    # Version-aware document construction
    # ---------------------------------------------------------

    print(
        "\nBuilding pooled "
        "version-aware corpus..."
    )

    corpus = build_hotpotqa_corpus(
        pooled_raw
    )

    print(
        f"Unique documents: "
        f"{len(corpus.documents):,}"
    )

    # ---------------------------------------------------------
    # Sentence-aware chunking
    # ---------------------------------------------------------

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
        f"Sentence chunks: "
        f"{len(chunks):,}"
    )

    # ---------------------------------------------------------
    # Structural checks
    #
    # These values freeze the exact corpus audited in 9.5A.
    # If they unexpectedly change, stop instead of silently
    # generating a different experimental corpus.
    # ---------------------------------------------------------

    assert len(pooled_raw) == 1800

    assert len(corpus.examples) == 1800

    assert len(corpus.documents) == 17403

    assert len(chunks) == 32368

    chunk_ids = [
        chunk.id
        for chunk in chunks
    ]

    assert (
        len(chunk_ids)
        == len(set(chunk_ids))
    ), "Duplicate chunk IDs detected."

    # ---------------------------------------------------------
    # Save chunks
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\nWriting chunks to "
        f"{CHUNKS_PATH}..."
    )

    with CHUNKS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        for chunk in chunks:

            record = {
                "id": chunk.id,
                "document_id": (
                    chunk.document_id
                ),
                "chunk_index": (
                    chunk.chunk_index
                ),
                "text": chunk.text,
                "start_char": (
                    chunk.start_char
                ),
                "end_char": (
                    chunk.end_char
                ),
                "metadata": (
                    chunk.metadata
                ),
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )

            file.write("\n")

    # ---------------------------------------------------------
    # Save manifest
    # ---------------------------------------------------------

    manifest = {
        "dataset": (
            "hotpotqa/hotpot_qa"
        ),
        "configuration": "distractor",

        "query_splits": {
            "exploratory": {
                "source": "train",
                "start": 0,
                "end_exclusive": 100,
                "count": 100,
            },
            "detector_train": {
                "source": "train",
                "start": 100,
                "end_exclusive": 1100,
                "count": 1000,
            },
            "detector_validation": {
                "source": "train",
                "start": 1100,
                "end_exclusive": 1300,
                "count": 200,
            },
            "detector_test": {
                "source": "validation",
                "start": 0,
                "end_exclusive": 500,
                "count": 500,
            },
        },

        "pooled_question_count": (
            len(corpus.examples)
        ),

        "unique_document_count": (
            len(corpus.documents)
        ),

        "sentence_chunk_count": (
            len(chunks)
        ),

        "sentence_chunking": {
            "sentences_per_chunk": 3,
            "sentence_overlap": 1,
        },

        "notes": [
            (
                "This is a pooled HotpotQA "
                "benchmark-derived retrieval corpus."
            ),
            (
                "It is not an open-Wikipedia "
                "retrieval corpus."
            ),
            (
                "Gold supporting facts are not "
                "stored as runtime retrieval features."
            ),
            (
                "The first 100 training examples "
                "remain the exploratory development "
                "slice used in earlier phases."
            ),
        ],
    }

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ---------------------------------------------------------
    # Verify persisted JSONL
    # ---------------------------------------------------------

    print(
        "\nVerifying persisted chunks..."
    )

    persisted_count = 0
    persisted_ids: set[str] = set()

    with CHUNKS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            record = json.loads(
                line
            )

            persisted_count += 1

            persisted_ids.add(
                record["id"]
            )

    assert (
        persisted_count
        == len(chunks)
    )

    assert (
        len(persisted_ids)
        == len(chunks)
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print("FIXED CORPUS SUMMARY")
    print("=" * 90)

    print(
        f"\nQuestions: "
        f"{len(corpus.examples):,}"
    )

    print(
        f"Documents: "
        f"{len(corpus.documents):,}"
    )

    print(
        f"Chunks: "
        f"{len(chunks):,}"
    )

    print(
        f"\nChunks artifact: "
        f"{CHUNKS_PATH}"
    )

    print(
        f"Manifest: "
        f"{MANIFEST_PATH}"
    )

    print(
        "\nPASS: Fixed scaled diagnostic "
        "corpus built and verified."
    )


if __name__ == "__main__":
    main()