from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """
    Standard internal representation of a source document.

    Every loader converts raw source files into this common format so
    downstream components do not need to know how the original file
    was stored.
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate the document immediately after creation.
        """

        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Document id must be a non-empty string.")

        if not isinstance(self.text, str):
            raise TypeError("Document text must be a string.")

        if not isinstance(self.metadata, dict):
            raise TypeError("Document metadata must be a dictionary.")

        self.id = self.id.strip()
        self.text = self.text.strip()

    @property
    def character_count(self) -> int:
        """Return the number of characters in the document."""
        return len(self.text)

    @property
    def word_count(self) -> int:
        """Return a simple whitespace-based word count."""
        return len(self.text.split())

    def is_empty(self) -> bool:
        """Return True when the document contains no usable text."""
        return not bool(self.text.strip())

    def to_dict(self) -> dict[str, Any]:
        """Convert the Document into a serializable dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
        }