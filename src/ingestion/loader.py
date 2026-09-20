import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.ingestion.document import Document


class BaseDocumentLoader(ABC):
    """
    Abstract interface that every document loader must follow.
    """

    @abstractmethod
    def load(self, file_path: Path) -> list[Document]:
        """
        Load a source file and return standardized Document objects.
        """
        raise NotImplementedError


class TextDocumentLoader(BaseDocumentLoader):
    """
    Loader for plain-text (.txt) documents.
    """

    def load(self, file_path: Path) -> list[Document]:
        file_path = Path(file_path)

        text = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        document = Document(
            id=file_path.stem,
            text=text,
            metadata={
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": file_path.suffix.lower(),
            },
        )

        return [document]


class JSONDocumentLoader(BaseDocumentLoader):
    """
    Loader for JSON documents.

    Supported structures:

    1. Single dictionary:
       {
           "id": "...",
           "text": "...",
           "metadata": {...}
       }

    2. List of dictionaries:
       [
           {
               "id": "...",
               "text": "...",
               "metadata": {...}
           }
       ]
    """

    def load(self, file_path: Path) -> list[Document]:
        file_path = Path(file_path)

        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if isinstance(data, dict):
            records = [data]

        elif isinstance(data, list):
            records = data

        else:
            raise ValueError(
                f"Unsupported JSON structure in {file_path}. "
                "Expected a dictionary or list of dictionaries."
            )

        documents: list[Document] = []

        for index, record in enumerate(records):

            if not isinstance(record, dict):
                raise ValueError(
                    f"JSON record {index} in {file_path} "
                    "is not a dictionary."
                )

            text = record.get("text")

            if text is None:
                raise ValueError(
                    f"JSON record {index} in {file_path} "
                    "does not contain a 'text' field."
                )

            document_id = str(
                record.get(
                    "id",
                    f"{file_path.stem}_{index}_{uuid4().hex[:8]}",
                )
            )

            metadata = record.get("metadata", {})

            if not isinstance(metadata, dict):
                raise ValueError(
                    f"'metadata' for record {index} "
                    "must be a dictionary."
                )

            # Add source-level metadata while preserving supplied metadata.
            metadata = {
                **metadata,
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": file_path.suffix.lower(),
            }

            document = Document(
                id=document_id,
                text=str(text),
                metadata=metadata,
            )

            documents.append(document)

        return documents


class DocumentLoader:
    """
    Main ingestion entry point.

    Determines which loader should process a file based on its extension.
    """

    def __init__(self) -> None:

        self.loaders: dict[str, BaseDocumentLoader] = {
            ".txt": TextDocumentLoader(),
            ".json": JSONDocumentLoader(),
        }

    def load_file(self, file_path: str | Path) -> list[Document]:

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File does not exist: {file_path}"
            )

        if not file_path.is_file():
            raise ValueError(
                f"Expected a file but received: {file_path}"
            )

        extension = file_path.suffix.lower()

        loader = self.loaders.get(extension)

        if loader is None:
            raise ValueError(
                f"Unsupported file type: '{extension}'. "
                f"Supported types: {sorted(self.loaders.keys())}"
            )

        documents = loader.load(file_path)

        # Remove documents that contain no usable text.
        documents = [
            document
            for document in documents
            if not document.is_empty()
        ]

        return documents

    def load_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
    ) -> list[Document]:

        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(
                f"Directory does not exist: {directory}"
            )

        if not directory.is_dir():
            raise ValueError(
                f"Expected a directory but received: {directory}"
            )

        documents: list[Document] = []

        pattern = "**/*" if recursive else "*"

        for file_path in sorted(directory.glob(pattern)):

            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in self.loaders:
                continue

            try:
                loaded_documents = self.load_file(file_path)
                documents.extend(loaded_documents)

            except Exception as error:
                print(
                    f"[WARNING] Could not load "
                    f"{file_path}: {error}"
                )

        return documents


def summarize_documents(
    documents: list[Document],
) -> dict[str, Any]:
    """
    Produce basic ingestion statistics.
    """

    total_characters = sum(
        document.character_count
        for document in documents
    )

    total_words = sum(
        document.word_count
        for document in documents
    )

    file_types: dict[str, int] = {}

    for document in documents:
        file_type = document.metadata.get(
            "file_type",
            "unknown",
        )

        file_types[file_type] = (
            file_types.get(file_type, 0) + 1
        )

    return {
        "document_count": len(documents),
        "total_characters": total_characters,
        "total_words": total_words,
        "file_types": file_types,
    }