from datasets import load_dataset

from src.ingestion.hotpotqa import (
    build_hotpotqa_corpus,
    validate_gold_evidence,
    find_invalid_gold_evidence,
)


DEVELOPMENT_EXAMPLES = 1000


def main() -> None:
    print("=" * 70)
    print("ADAPTIVERAG - HOTPOTQA ADAPTER TEST")
    print("=" * 70)

    # --------------------------------------------------
    # 1. LOAD HOTPOTQA
    # --------------------------------------------------

    print("\nLoading HotpotQA distractor configuration...")

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    train_dataset = dataset["train"]

    print(
        f"Full training examples: "
        f"{len(train_dataset):,}"
    )

    # --------------------------------------------------
    # 2. BUILD DEVELOPMENT CORPUS
    # --------------------------------------------------

    print(
        f"\nBuilding development corpus from "
        f"{DEVELOPMENT_EXAMPLES:,} examples..."
    )

    corpus = build_hotpotqa_corpus(
        train_dataset,
        max_examples=DEVELOPMENT_EXAMPLES,
    )

    # --------------------------------------------------
    # 3. CORPUS STATISTICS
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("CORPUS STATISTICS")
    print("=" * 70)

    print(
        f"QA examples: "
        f"{len(corpus.examples):,}"
    )

    print(
        f"Unique document versions: "
        f"{len(corpus.documents):,}"
    )

    total_characters = sum(
        document.character_count
        for document in corpus.documents
    )

    total_words = sum(
        document.word_count
        for document in corpus.documents
    )

    print(
        f"Total document characters: "
        f"{total_characters:,}"
    )

    print(
        f"Total document words: "
        f"{total_words:,}"
    )

    if corpus.documents:
        average_words = (
            total_words
            / len(corpus.documents)
        )

        print(
            f"Average words/document: "
            f"{average_words:.2f}"
        )

    # --------------------------------------------------
    # 4. VALIDATE GOLD EVIDENCE
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("GOLD EVIDENCE VALIDATION")
    print("=" * 70)

    validation = validate_gold_evidence(
        corpus
    )

    for key, value in validation.items():
        print(
            f"{key}: {value:,}"
        )

    # --------------------------------------------------
    # 5. INVESTIGATE INVALID GOLD EVIDENCE
    # --------------------------------------------------

    problems = find_invalid_gold_evidence(
        corpus
    )

    print("\n" + "=" * 70)
    print("GOLD EVIDENCE PROBLEMS")
    print("=" * 70)

    if problems:

        print(
            f"\nProblems found: "
            f"{len(problems):,}"
        )

        for problem_index, problem in enumerate(
            problems,
            start=1,
        ):
            print("\n" + "-" * 70)

            print(
                f"Problem #{problem_index}"
            )

            for key, value in problem.items():
                print(
                    f"{key}: {value}"
                )

    else:
        print(
            "\nNo gold evidence problems found."
        )

    # --------------------------------------------------
    # 6. DISPLAY FIRST QA EXAMPLE
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("FIRST QA EXAMPLE")
    print("=" * 70)

    example = corpus.examples[0]

    print(
        f"ID: {example.id}"
    )

    print(
        f"Question: {example.question}"
    )

    print(
        f"Answer: {example.answer}"
    )

    print(
        f"Type: {example.question_type}"
    )

    print(
        f"Level: {example.level}"
    )

    print("\nGold supporting facts:")

    for fact in example.supporting_facts:
        print(
            f"  Document ID: {fact.document_id} | "
            f"Title: {fact.title} | "
            f"Sentence ID: {fact.sentence_id}"
        )

    # --------------------------------------------------
    # 7. RESOLVE GOLD EVIDENCE
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("RESOLVED GOLD EVIDENCE")
    print("=" * 70)

    documents_by_id = {
        document.id: document
        for document in corpus.documents
    }

    for fact in example.supporting_facts:

        document = documents_by_id.get(
            fact.document_id
        )

        if document is None:
            print(
                f"\nMISSING DOCUMENT: "
                f"{fact.document_id}"
            )
            continue

        sentences = document.metadata[
            "sentences"
        ]

        if (
            fact.sentence_id < 0
            or fact.sentence_id >= len(sentences)
        ):
            print(
                f"\nINVALID SENTENCE ID: "
                f"{fact.title} "
                f"[{fact.sentence_id}]"
            )
            continue

        print(
            f"\nDocument ID: "
            f"{fact.document_id}"
        )

        print(
            f"Document: "
            f"{fact.title}"
        )

        print(
            f"Sentence ID: "
            f"{fact.sentence_id}"
        )

        print(
            f"Evidence: "
            f"{sentences[fact.sentence_id]}"
        )

    # --------------------------------------------------
    # 8. DISPLAY FIRST CORPUS DOCUMENT
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("FIRST CORPUS DOCUMENT")
    print("=" * 70)

    document = corpus.documents[0]

    print(
        f"Document ID: "
        f"{document.id}"
    )

    print(
        f"Title: "
        f"{document.metadata['title']}"
    )

    print(
        f"Sentence count: "
        f"{document.metadata['sentence_count']}"
    )

    print(
        f"Characters: "
        f"{document.character_count}"
    )

    print(
        f"Words: "
        f"{document.word_count}"
    )

    print("\nIndexed sentences:")

    for sentence_index, sentence in enumerate(
        document.metadata["sentences"]
    ):
        print(
            f"  [{sentence_index}] "
            f"{repr(sentence)}"
        )

    print("\nFlattened retrieval text:")
    print(
        document.text
    )

    # --------------------------------------------------
    # 9. COMPLETE
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("ADAPTER TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()