from dataclasses import dataclass
from typing import Iterable

from src.chunking.chunk import Chunk
from src.ingestion.document import Document


@dataclass(frozen=True)
class SentenceChunkingConfig:
    """
    Configuration for sentence-aware chunking.

    sentences_per_chunk:
        Maximum number of original sentence positions in one chunk.

    sentence_overlap:
        Number of sentence positions shared between consecutive chunks.
    """

    sentences_per_chunk: int = 3
    sentence_overlap: int = 1

    def __post_init__(self) -> None:
        if self.sentences_per_chunk <= 0:
            raise ValueError(
                "sentences_per_chunk must be greater than 0."
            )

        if self.sentence_overlap < 0:
            raise ValueError(
                "sentence_overlap cannot be negative."
            )

        if self.sentence_overlap >= self.sentences_per_chunk:
            raise ValueError(
                "sentence_overlap must be smaller than "
                "sentences_per_chunk."
            )


class SentenceChunker:
    """
    Sentence-aware chunker for documents containing an ordered
    metadata['sentences'] list.

    Original sentence positions are preserved so evidence represented
    as (document_id, sentence_id) can later be mapped exactly to
    retrieval chunks.
    """

    def __init__(
        self,
        config: SentenceChunkingConfig | None = None,
    ) -> None:
        self.config = config or SentenceChunkingConfig()

    def chunk_document(
        self,
        document: Document,
    ) -> list[Chunk]:
        sentences = document.metadata.get("sentences")

        if not isinstance(sentences, list):
            raise ValueError(
                f"Document '{document.id}' does not contain "
                "metadata['sentences'] as a list."
            )

        if not sentences:
            return []

        normalized_sentences = [
            sentence.strip()
            if isinstance(sentence, str)
            else ""
            for sentence in sentences
        ]

        chunks: list[Chunk] = []

        step = (
            self.config.sentences_per_chunk
            - self.config.sentence_overlap
        )

        chunk_index = 0
        start_sentence = 0

        while start_sentence < len(normalized_sentences):
            end_sentence = min(
                start_sentence
                + self.config.sentences_per_chunk,
                len(normalized_sentences),
            )

            sentence_ids = list(
                range(
                    start_sentence,
                    end_sentence,
                )
            )

            window_sentences = normalized_sentences[
                start_sentence:end_sentence
            ]

            non_empty_sentences = [
                sentence
                for sentence in window_sentences
                if sentence
            ]

            # Preserve original sentence positions, but do not emit
            # retrieval chunks that contain no usable text.
            if non_empty_sentences:
                chunk_text = " ".join(
                    non_empty_sentences
                )

                chunk_id = (
                    f"{document.id}"
                    f"_sentence_chunk_"
                    f"{chunk_index:05d}"
                )

                # Do not duplicate the entire source sentence list
                # inside every chunk. The relevant sentence IDs are
                # already stored explicitly below.
                chunk_metadata = {
                    key: value
                    for key, value
                    in document.metadata.items()
                    if key != "sentences"
                }

                chunk_metadata.update(
                    {
                        "document_id": document.id,
                        "chunk_index": chunk_index,
                        "chunking_strategy": "sentence",
                        "sentence_start": start_sentence,
                        "sentence_end": end_sentence - 1,
                        "sentence_ids": sentence_ids,
                    }
                )

                chunks.append(
                    Chunk(
                        id=chunk_id,
                        document_id=document.id,
                        text=chunk_text,
                        chunk_index=chunk_index,
                        start_char=0,
                        end_char=len(chunk_text),
                        metadata=chunk_metadata,
                    )
                )

                chunk_index += 1

            if end_sentence == len(
                normalized_sentences
            ):
                break

            start_sentence += step

        return chunks

    def chunk_documents(
        self,
        documents: Iterable[Document],
    ) -> list[Chunk]:
        chunks: list[Chunk] = []

        for document in documents:
            chunks.extend(
                self.chunk_document(document)
            )

        return chunks


def find_chunks_for_evidence(
    chunks: Iterable[Chunk],
    document_id: str,
    sentence_id: int,
) -> list[Chunk]:
    """
    Return every chunk containing the requested sentence-level
    evidence.

    This helper is convenient for small tests. For large-scale
    evaluation, build_evidence_chunk_index() should be used instead
    of repeatedly scanning every chunk.
    """

    matches: list[Chunk] = []

    for chunk in chunks:
        if chunk.document_id != document_id:
            continue

        sentence_ids = chunk.metadata.get(
            "sentence_ids",
            [],
        )

        if sentence_id in sentence_ids:
            matches.append(chunk)

    return matches


def build_evidence_chunk_index(
    chunks: Iterable[Chunk],
) -> dict[tuple[str, int], list[str]]:
    """
    Build a lookup from sentence-level evidence identity to the
    chunk IDs containing that evidence.

    Key:
        (document_id, sentence_id)

    Value:
        list of chunk IDs containing that sentence position.

    Example:

        (
            "hotpotqa_abc123",
            2,
        )
            ->
        [
            "hotpotqa_abc123_sentence_chunk_00000",
            "hotpotqa_abc123_sentence_chunk_00001",
        ]

    Overlapping sentence windows can therefore correctly map one
    source sentence to multiple retrieval chunks.
    """

    evidence_index: dict[
        tuple[str, int],
        list[str],
    ] = {}

    for chunk in chunks:
        sentence_ids = chunk.metadata.get(
            "sentence_ids",
            [],
        )

        for sentence_id in sentence_ids:
            key = (
                chunk.document_id,
                sentence_id,
            )

            evidence_index.setdefault(
                key,
                [],
            ).append(chunk.id)

    return evidence_index