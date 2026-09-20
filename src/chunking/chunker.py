from dataclasses import dataclass

from src.chunking.chunk import Chunk
from src.ingestion.document import Document


@dataclass
class ChunkingConfig:
    """
    Configuration for character-based overlapping chunking.
    """

    chunk_size: int = 500
    chunk_overlap: int = 100

    def __post_init__(self) -> None:

        if self.chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )

        if self.chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )


class CharacterChunker:
    """
    Split documents into fixed-size overlapping character chunks.

    This is intentionally our first chunking baseline.

    Later experiments can compare this baseline against:
    - token-based chunking
    - sentence-aware chunking
    - recursive chunking
    - semantic chunking
    """

    def __init__(
        self,
        config: ChunkingConfig | None = None,
    ) -> None:

        self.config = config or ChunkingConfig()

    def chunk_document(
        self,
        document: Document,
    ) -> list[Chunk]:

        if document.is_empty():
            return []

        text = document.text

        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        step_size = chunk_size - overlap

        chunks: list[Chunk] = []

        start = 0
        chunk_index = 0

        while start < len(text):

            end = min(
                start + chunk_size,
                len(text),
            )

            chunk_text = text[start:end]

            if chunk_text.strip():

                chunk_id = (
                    f"{document.id}_chunk_"
                    f"{chunk_index:05d}"
                )

                chunk_metadata = {
                    **document.metadata,
                    "document_id": document.id,
                    "chunk_index": chunk_index,
                }

                chunk = Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    metadata=chunk_metadata,
                )

                chunks.append(chunk)

                chunk_index += 1

            if end >= len(text):
                break

            start += step_size

        return chunks

    def chunk_documents(
        self,
        documents: list[Document],
    ) -> list[Chunk]:

        chunks: list[Chunk] = []

        for document in documents:

            document_chunks = self.chunk_document(
                document
            )

            chunks.extend(document_chunks)

        return chunks


def summarize_chunks(
    chunks: list[Chunk],
) -> dict:

    if not chunks:
        return {
            "chunk_count": 0,
            "document_count": 0,
            "average_characters": 0.0,
            "average_words": 0.0,
            "min_characters": 0,
            "max_characters": 0,
        }

    character_counts = [
        chunk.character_count
        for chunk in chunks
    ]

    word_counts = [
        chunk.word_count
        for chunk in chunks
    ]

    document_ids = {
        chunk.document_id
        for chunk in chunks
    }

    return {
        "chunk_count": len(chunks),
        "document_count": len(document_ids),
        "average_characters": (
            sum(character_counts)
            / len(character_counts)
        ),
        "average_words": (
            sum(word_counts)
            / len(word_counts)
        ),
        "min_characters": min(character_counts),
        "max_characters": max(character_counts),
    }