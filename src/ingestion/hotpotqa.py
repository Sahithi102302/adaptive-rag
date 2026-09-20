from dataclasses import dataclass
from typing import Any
import hashlib

from datasets import Dataset

from src.ingestion.document import Document


@dataclass(frozen=True)
class SupportingFact:
    """
    One gold supporting fact from a HotpotQA question.

    document_id identifies the exact document/content version that
    appeared in this question's context.

    title and sentence_id preserve the original HotpotQA annotation.
    """

    document_id: str
    title: str
    sentence_id: int


@dataclass
class HotpotQAExample:
    """
    Internal representation of one HotpotQA QA example.
    """

    id: str
    question: str
    answer: str
    question_type: str
    level: str
    supporting_facts: list[SupportingFact]


@dataclass
class HotpotQACorpus:
    """
    Adapted HotpotQA corpus.

    documents:
        unique document CONTENT versions available for retrieval

    examples:
        questions with gold evidence mapped to exact document versions
    """

    documents: list[Document]
    examples: list[HotpotQAExample]


def normalize_title(title: str) -> str:
    """
    Conservatively normalize a document title.
    """

    return " ".join(
        title.strip().split()
    )


def normalize_sentences(
    sentences: list[str],
) -> list[str]:
    """
    Normalize sentence text while preserving every original
    sentence position.

    Empty sentences are NOT removed because HotpotQA supporting
    facts refer to sentences using positional sentence IDs.
    """

    return [
        sentence.strip()
        if isinstance(sentence, str)
        else ""
        for sentence in sentences
    ]


def make_document_id(
    title: str,
    sentences: list[str],
) -> str:
    """
    Create a stable identifier for an exact document version.

    Title alone is insufficient because the full HotpotQA audit
    showed that some normalized titles occur with multiple distinct
    content versions.

    Identity therefore depends on:

        normalized title + normalized sentence content
    """

    normalized_title = normalize_title(
        title
    )

    normalized_sentence_list = (
        normalize_sentences(
            sentences
        )
    )

    content = "\n".join(
        normalized_sentence_list
    )

    identity_material = (
        normalized_title
        + "\n"
        + content
    )

    digest = hashlib.sha256(
        identity_material.encode("utf-8")
    ).hexdigest()[:16]

    return f"hotpotqa_{digest}"


def build_document(
    title: str,
    sentences: list[str],
) -> Document:
    """
    Convert one HotpotQA context document into our standard
    Document representation.

    Sentence positions are preserved in metadata.

    The flattened text excludes empty strings because they contain
    no retrievable information, while metadata retains those empty
    positions so HotpotQA sentence IDs remain valid.
    """

    normalized_title = normalize_title(
        title
    )

    preserved_sentences = (
        normalize_sentences(
            sentences
        )
    )

    text = " ".join(
        sentence
        for sentence in preserved_sentences
        if sentence
    )

    document_id = make_document_id(
        normalized_title,
        preserved_sentences,
    )

    return Document(
        id=document_id,
        text=text,
        metadata={
            "dataset": "hotpotqa",
            "title": normalized_title,
            "sentences": preserved_sentences,
            "sentence_count": len(
                preserved_sentences
            ),
        },
    )


def build_hotpotqa_corpus(
    dataset: Dataset,
    max_examples: int | None = None,
) -> HotpotQACorpus:
    """
    Convert a HotpotQA dataset split into:

        1. a deduplicated retrieval corpus
        2. QA/evaluation examples
        3. gold evidence linked to exact document versions

    Documents are deduplicated by stable document ID, which is
    derived from both title and content.
    """

    if max_examples is not None:

        if max_examples <= 0:
            raise ValueError(
                "max_examples must be greater than zero."
            )

        dataset = dataset.select(
            range(
                min(
                    max_examples,
                    len(dataset),
                )
            )
        )

    documents_by_id: dict[
        str,
        Document,
    ] = {}

    examples: list[
        HotpotQAExample
    ] = []

    for record in dataset:

        context_titles = (
            record["context"]["title"]
        )

        context_sentence_groups = (
            record["context"]["sentences"]
        )

        # --------------------------------------------------
        # Build exact document versions for THIS question.
        # --------------------------------------------------

        local_documents_by_title: dict[
            str,
            Document,
        ] = {}

        for title, sentences in zip(
            context_titles,
            context_sentence_groups,
        ):

            document = build_document(
                title=title,
                sentences=sentences,
            )

            documents_by_id[
                document.id
            ] = document

            normalized_title = (
                document.metadata["title"]
            )

            local_documents_by_title[
                normalized_title
            ] = document

        # --------------------------------------------------
        # Resolve gold evidence against THIS question's
        # context so that title ambiguity across the global
        # corpus does not destroy document identity.
        # --------------------------------------------------

        supporting_titles = (
            record["supporting_facts"]["title"]
        )

        supporting_sentence_ids = (
            record["supporting_facts"]["sent_id"]
        )

        supporting_facts: list[
            SupportingFact
        ] = []

        for title, sentence_id in zip(
            supporting_titles,
            supporting_sentence_ids,
        ):

            normalized_title = (
                normalize_title(
                    title
                )
            )

            local_document = (
                local_documents_by_title.get(
                    normalized_title
                )
            )

            # The full audit found no missing supporting
            # documents, but we fail loudly if one appears
            # in another split or future dataset version.
            if local_document is None:
                raise ValueError(
                    "Supporting document missing "
                    "from question context. "
                    f"Example ID: {record['id']}, "
                    f"title: {normalized_title}"
                )

            supporting_facts.append(
                SupportingFact(
                    document_id=local_document.id,
                    title=normalized_title,
                    sentence_id=int(
                        sentence_id
                    ),
                )
            )

        example = HotpotQAExample(
            id=record["id"],
            question=record[
                "question"
            ].strip(),
            answer=record[
                "answer"
            ].strip(),
            question_type=record["type"],
            level=record["level"],
            supporting_facts=supporting_facts,
        )

        examples.append(
            example
        )

    return HotpotQACorpus(
        documents=list(
            documents_by_id.values()
        ),
        examples=examples,
    )


def validate_gold_evidence(
    corpus: HotpotQACorpus,
) -> dict[str, int]:
    """
    Validate gold evidence against exact document versions.

    Invalid source annotations are reported rather than silently
    repaired or deleted.
    """

    documents_by_id = {
        document.id: document
        for document in corpus.documents
    }

    total_supporting_facts = 0
    valid_supporting_facts = 0
    missing_documents = 0
    invalid_sentence_ids = 0
    empty_gold_sentences = 0

    for example in corpus.examples:

        for fact in example.supporting_facts:

            total_supporting_facts += 1

            document = documents_by_id.get(
                fact.document_id
            )

            if document is None:
                missing_documents += 1
                continue

            sentences = document.metadata[
                "sentences"
            ]

            if (
                fact.sentence_id < 0
                or fact.sentence_id
                >= len(sentences)
            ):
                invalid_sentence_ids += 1
                continue

            if not sentences[
                fact.sentence_id
            ]:
                empty_gold_sentences += 1
                continue

            valid_supporting_facts += 1

    return {
        "total_supporting_facts":
            total_supporting_facts,

        "valid_supporting_facts":
            valid_supporting_facts,

        "missing_documents":
            missing_documents,

        "invalid_sentence_ids":
            invalid_sentence_ids,

        "empty_gold_sentences":
            empty_gold_sentences,
    }


def find_invalid_gold_evidence(
    corpus: HotpotQACorpus,
) -> list[dict[str, Any]]:
    """
    Return detailed information about malformed gold evidence.

    This is a diagnostic function. It does not modify or repair
    benchmark annotations.
    """

    documents_by_id = {
        document.id: document
        for document in corpus.documents
    }

    problems: list[
        dict[str, Any]
    ] = []

    for example in corpus.examples:

        for fact in example.supporting_facts:

            document = documents_by_id.get(
                fact.document_id
            )

            if document is None:

                problems.append(
                    {
                        "example_id":
                            example.id,
                        "question":
                            example.question,
                        "document_id":
                            fact.document_id,
                        "title":
                            fact.title,
                        "sentence_id":
                            fact.sentence_id,
                        "problem":
                            "missing_document",
                    }
                )

                continue

            sentences = document.metadata[
                "sentences"
            ]

            if (
                fact.sentence_id < 0
                or fact.sentence_id
                >= len(sentences)
            ):

                problems.append(
                    {
                        "example_id":
                            example.id,
                        "question":
                            example.question,
                        "document_id":
                            fact.document_id,
                        "title":
                            fact.title,
                        "sentence_id":
                            fact.sentence_id,
                        "stored_sentence_count":
                            len(sentences),
                        "problem":
                            "invalid_sentence_id",
                    }
                )

                continue

            if not sentences[
                fact.sentence_id
            ]:

                problems.append(
                    {
                        "example_id":
                            example.id,
                        "question":
                            example.question,
                        "document_id":
                            fact.document_id,
                        "title":
                            fact.title,
                        "sentence_id":
                            fact.sentence_id,
                        "problem":
                            "empty_gold_sentence",
                    }
                )

    return problems