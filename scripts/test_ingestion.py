from pathlib import Path

from src.ingestion.loader import (
    DocumentLoader,
    summarize_documents,
)


def main() -> None:

    project_root = Path(__file__).resolve().parents[1]

    demo_data_directory = (
        project_root
        / "data"
        / "demo"
    )

    loader = DocumentLoader()

    documents = loader.load_directory(
        demo_data_directory
    )

    print("=" * 70)
    print("ADAPTIVERAG - INGESTION TEST")
    print("=" * 70)

    print(f"\nLoaded documents: {len(documents)}")

    for index, document in enumerate(
        documents,
        start=1,
    ):
        print("\n" + "-" * 70)

        print(f"Document #{index}")
        print(f"ID: {document.id}")
        print(
            f"Characters: "
            f"{document.character_count}"
        )
        print(
            f"Words: "
            f"{document.word_count}"
        )

        print("Metadata:")

        for key, value in document.metadata.items():
            print(f"  {key}: {value}")

        preview = document.text[:200]

        print(f"Text preview: {preview}")

    summary = summarize_documents(
        documents
    )

    print("\n" + "=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)

    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()