from datasets import load_dataset

from src.chunking.sentence_chunker import (
    SentenceChunker,
    SentenceChunkingConfig,
)
from src.ingestion.hotpotqa import (
    build_hotpotqa_corpus,
)



TARGET_TITLE = "Arthur's Magazine"

def format_chunk_for_trace(chunk) -> str:
    """
    Reproduce the retrieval-text formatting without importing
    the dense retrieval / embedding stack.
    """

    title = chunk.metadata.get("title")

    if isinstance(title, str) and title.strip():
        return f"{title.strip()}: {chunk.text}"

    return chunk.text

def print_separator(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def main() -> None:
    print_separator(
        "HotpotQA Text Transformation Trace"
    )

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    raw_example = dataset["train"][0]

    print_separator("1. RAW HOTPOTQA CONTEXT")

    raw_titles = raw_example["context"]["title"]
    raw_sentences = raw_example["context"]["sentences"]

    target_raw_sentences = None

    for title, sentences in zip(
        raw_titles,
        raw_sentences,
    ):
        if title == TARGET_TITLE:
            target_raw_sentences = sentences
            break

    if target_raw_sentences is None:
        raise RuntimeError(
            f"Could not find {TARGET_TITLE!r} "
            "in raw example."
        )

    print(f"Title: {TARGET_TITLE!r}")

    for sentence_id, sentence in enumerate(
        target_raw_sentences
    ):
        print(
            f"\nSentence {sentence_id}:"
        )
        print(
            f"repr: {sentence!r}"
        )
        print(
            f"text: {sentence}"
        )

    print_separator(
        "2. AFTER HOTPOTQA ADAPTER"
    )

    corpus = build_hotpotqa_corpus(
        dataset["train"],
        max_examples=1,
    )

    example = corpus.examples[0]

    target_document = None

    for document in corpus.documents:
        title = document.metadata.get("title")

        if title == TARGET_TITLE:
            target_document = document
            break

    if target_document is None:
        raise RuntimeError(
            f"Could not find {TARGET_TITLE!r} "
            "in adapted corpus."
        )

    print(
        f"Document ID: "
        f"{target_document.id}"
    )

    print(
        f"Metadata title: "
        f"{target_document.metadata.get('title')!r}"
    )

    adapted_sentences = (
        target_document.metadata["sentences"]
    )

    for sentence_id, sentence in enumerate(
        adapted_sentences
    ):
        print(
            f"\nSentence {sentence_id}:"
        )
        print(
            f"repr: {sentence!r}"
        )
        print(
            f"text: {sentence}"
        )

    print_separator(
        "3. FLATTENED DOCUMENT.TEXT"
    )

    print(
        repr(target_document.text)
    )

    print("\nRendered:")
    print(target_document.text)

    print_separator(
        "4. SENTENCE-AWARE CHUNKS"
    )

    chunker = SentenceChunker(
        SentenceChunkingConfig(
            sentences_per_chunk=3,
            sentence_overlap=1,
        )
    )

    chunks = chunker.chunk_document(
        target_document
    )

    for chunk in chunks:
        print(
            f"\nChunk: {chunk.id}"
        )

        print(
            f"Sentence IDs: "
            f"{chunk.metadata['sentence_ids']}"
        )

        print(
            f"repr: {chunk.text!r}"
        )

        print(
            f"text: {chunk.text}"
        )

    print_separator(
        "5. DENSE RETRIEVAL REPRESENTATION"
    )

    for chunk in chunks:
        retrieval_text = (
            format_chunk_for_trace(
                chunk
            )
        )

        print(
            f"\nChunk: {chunk.id}"
        )

        print(
            f"repr: {retrieval_text!r}"
        )

        print(
            f"text: {retrieval_text}"
        )

    print_separator(
        "6. GOLD SUPPORTING FACTS"
    )

    for fact in example.supporting_facts:
        if fact.title == TARGET_TITLE:
            sentence = adapted_sentences[
                fact.sentence_id
            ]

            print(
                f"Document ID: "
                f"{fact.document_id}"
            )

            print(
                f"Sentence ID: "
                f"{fact.sentence_id}"
            )

            print(
                f"Sentence repr: "
                f"{sentence!r}"
            )

            print(
                f"Sentence text: "
                f"{sentence}"
            )

    print_separator(
        "TRACE COMPLETE"
    )

    print(
        "No text was modified by this script."
    )


if __name__ == "__main__":
    main()