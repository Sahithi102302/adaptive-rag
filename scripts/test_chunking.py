from pathlib import Path

from src.chunking.chunker import (
    CharacterChunker,
    ChunkingConfig,
    summarize_chunks,
)
from src.ingestion.loader import DocumentLoader


def main() -> None:

    project_root = Path(__file__).resolve().parents[1]

    demo_data_directory = (
        project_root
        / "data"
        / "demo"
    )

    # --------------------------------------------------
    # 1. INGEST DOCUMENTS
    # --------------------------------------------------

    loader = DocumentLoader()

    documents = loader.load_directory(
        demo_data_directory
    )

    print("=" * 70)
    print("ADAPTIVERAG - CHUNKING TEST")
    print("=" * 70)

    print(
        f"\nLoaded source documents: "
        f"{len(documents)}"
    )

    # --------------------------------------------------
    # 2. CONFIGURE CHUNKER
    # --------------------------------------------------

    config = ChunkingConfig(
        chunk_size=60,
        chunk_overlap=15,
    )

    print(
        f"Chunk size: {config.chunk_size}"
    )

    print(
        f"Chunk overlap: {config.chunk_overlap}"
    )

    print(
        f"Step size: "
        f"{config.chunk_size - config.chunk_overlap}"
    )

    # --------------------------------------------------
    # 3. CHUNK DOCUMENTS
    # --------------------------------------------------

    chunker = CharacterChunker(
        config=config
    )

    chunks = chunker.chunk_documents(
        documents
    )

    # --------------------------------------------------
    # 4. DISPLAY CHUNKS
    # --------------------------------------------------

    for chunk in chunks:

        print("\n" + "-" * 70)

        print(f"Chunk ID: {chunk.id}")
        print(
            f"Source document: "
            f"{chunk.document_id}"
        )

        print(
            f"Chunk index: "
            f"{chunk.chunk_index}"
        )

        print(
            f"Character range: "
            f"[{chunk.start_char}:{chunk.end_char}]"
        )

        print(
            f"Characters: "
            f"{chunk.character_count}"
        )

        print(
            f"Words: "
            f"{chunk.word_count}"
        )

        print("Metadata:")

        for key, value in chunk.metadata.items():
            print(f"  {key}: {value}")

        print("Text:")

        print(
            repr(chunk.text)
        )

    # --------------------------------------------------
    # 5. SUMMARY
    # --------------------------------------------------

    summary = summarize_chunks(
        chunks
    )

    print("\n" + "=" * 70)
    print("CHUNKING SUMMARY")
    print("=" * 70)

    for key, value in summary.items():

        if isinstance(value, float):
            print(
                f"{key}: {value:.2f}"
            )
        else:
            print(
                f"{key}: {value}"
            )


if __name__ == "__main__":
    main()