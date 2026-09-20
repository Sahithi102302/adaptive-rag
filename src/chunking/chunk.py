from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """
    Standard representation of a chunk derived from a source document.

    A Chunk preserves lineage back to the original document while
    containing the smaller text unit that will later be embedded,
    indexed, retrieved, reranked, and evaluated.
    """

    id: str
    document_id: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate and normalize the chunk after creation.
        """

        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Chunk id must be a non-empty string.")

        if (
            not isinstance(self.document_id, str)
            or not self.document_id.strip()
        ):
            raise ValueError(
                "Document id must be a non-empty string."
            )

        if not isinstance(self.text, str):
            raise TypeError("Chunk text must be a string.")

        if not isinstance(self.chunk_index, int):
            raise TypeError("Chunk index must be an integer.")

        if self.chunk_index < 0:
            raise ValueError(
                "Chunk index cannot be negative."
            )

        if not isinstance(self.start_char, int):
            raise TypeError(
                "start_char must be an integer."
            )

        if not isinstance(self.end_char, int):
            raise TypeError(
                "end_char must be an integer."
            )

        if self.start_char < 0:
            raise ValueError(
                "start_char cannot be negative."
            )

        if self.end_char < self.start_char:
            raise ValueError(
                "end_char cannot be smaller than start_char."
            )

        if not isinstance(self.metadata, dict):
            raise TypeError(
                "Chunk metadata must be a dictionary."
            )

        self.id = self.id.strip()
        self.document_id = self.document_id.strip()
        self.text = self.text.strip()

    @property
    def character_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def is_empty(self) -> bool:
        return not bool(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata,
        }