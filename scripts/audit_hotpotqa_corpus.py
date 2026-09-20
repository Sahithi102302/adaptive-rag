from collections import Counter, defaultdict
from hashlib import sha256

from datasets import load_dataset


def normalize_title(title: str) -> str:
    return " ".join(title.strip().split())


def normalize_sentences(
    sentences: list[str],
) -> tuple[str, ...]:
    """
    Preserve sentence positions while normalizing whitespace.
    """

    return tuple(
        sentence.strip()
        if isinstance(sentence, str)
        else ""
        for sentence in sentences
    )


def content_hash(
    sentences: tuple[str, ...],
) -> str:
    """
    Stable fingerprint for one document version.
    """

    content = "\n".join(sentences)

    return sha256(
        content.encode("utf-8")
    ).hexdigest()


def main() -> None:
    print("=" * 70)
    print("ADAPTIVERAG - FULL HOTPOTQA CORPUS AUDIT")
    print("=" * 70)

    # --------------------------------------------------
    # 1. LOAD FULL TRAINING SPLIT
    # --------------------------------------------------

    print("\nLoading HotpotQA distractor configuration...")

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    train_dataset = dataset["train"]

    print(
        f"Training examples: "
        f"{len(train_dataset):,}"
    )

    # --------------------------------------------------
    # 2. AUDIT STRUCTURES
    # --------------------------------------------------

    title_occurrences = Counter()

    title_content_hashes = defaultdict(set)

    title_versions = defaultdict(dict)

    empty_sentence_count = 0
    documents_with_empty_sentences = 0

    total_document_occurrences = 0
    total_supporting_facts = 0

    valid_supporting_facts = 0
    missing_supporting_documents = 0
    invalid_sentence_ids = 0
    empty_gold_sentences = 0

    malformed_examples = set()

    invalid_examples = []

    # --------------------------------------------------
    # 3. PROCESS ALL 90K+ QUESTIONS
    # --------------------------------------------------

    print("\nAuditing complete training split...")

    for example_index, record in enumerate(
        train_dataset
    ):
        context_titles = record["context"]["title"]
        context_sentence_groups = (
            record["context"]["sentences"]
        )

        # Map documents inside this individual example.
        local_documents = {}

        for title, sentences in zip(
            context_titles,
            context_sentence_groups,
        ):
            normalized_title = normalize_title(
                title
            )

            normalized_sentence_tuple = (
                normalize_sentences(
                    sentences
                )
            )

            total_document_occurrences += 1

            title_occurrences[
                normalized_title
            ] += 1

            document_hash = content_hash(
                normalized_sentence_tuple
            )

            title_content_hashes[
                normalized_title
            ].add(
                document_hash
            )

            if (
                document_hash
                not in title_versions[
                    normalized_title
                ]
            ):
                title_versions[
                    normalized_title
                ][document_hash] = (
                    normalized_sentence_tuple
                )

            local_documents[
                normalized_title
            ] = normalized_sentence_tuple

            empty_count = sum(
                1
                for sentence
                in normalized_sentence_tuple
                if not sentence
            )

            if empty_count > 0:
                documents_with_empty_sentences += 1
                empty_sentence_count += (
                    empty_count
                )

        # --------------------------------------------------
        # Validate gold supporting facts against the
        # ORIGINAL DOCUMENT VERSION IN THIS QUESTION.
        # --------------------------------------------------

        supporting_titles = (
            record["supporting_facts"]["title"]
        )

        supporting_sentence_ids = (
            record["supporting_facts"]["sent_id"]
        )

        for title, sentence_id in zip(
            supporting_titles,
            supporting_sentence_ids,
        ):
            total_supporting_facts += 1

            normalized_title = normalize_title(
                title
            )

            sentence_id = int(
                sentence_id
            )

            sentences = local_documents.get(
                normalized_title
            )

            if sentences is None:
                missing_supporting_documents += 1
                malformed_examples.add(
                    record["id"]
                )

                invalid_examples.append(
                    {
                        "example_index":
                            example_index,
                        "example_id":
                            record["id"],
                        "question":
                            record["question"],
                        "title":
                            normalized_title,
                        "sentence_id":
                            sentence_id,
                        "problem":
                            "missing_document",
                    }
                )

                continue

            if (
                sentence_id < 0
                or sentence_id >= len(sentences)
            ):
                invalid_sentence_ids += 1
                malformed_examples.add(
                    record["id"]
                )

                invalid_examples.append(
                    {
                        "example_index":
                            example_index,
                        "example_id":
                            record["id"],
                        "question":
                            record["question"],
                        "title":
                            normalized_title,
                        "sentence_id":
                            sentence_id,
                        "sentence_count":
                            len(sentences),
                        "problem":
                            "invalid_sentence_id",
                    }
                )

                continue

            if not sentences[sentence_id]:
                empty_gold_sentences += 1
                malformed_examples.add(
                    record["id"]
                )

                invalid_examples.append(
                    {
                        "example_index":
                            example_index,
                        "example_id":
                            record["id"],
                        "question":
                            record["question"],
                        "title":
                            normalized_title,
                        "sentence_id":
                            sentence_id,
                        "problem":
                            "empty_gold_sentence",
                    }
                )

                continue

            valid_supporting_facts += 1

        if (
            (example_index + 1) % 10000
            == 0
        ):
            print(
                f"Processed "
                f"{example_index + 1:,} / "
                f"{len(train_dataset):,} "
                f"examples..."
            )

    # --------------------------------------------------
    # 4. TITLE / CONTENT ANALYSIS
    # --------------------------------------------------

    repeated_titles = {
        title: count
        for title, count
        in title_occurrences.items()
        if count > 1
    }

    conflicting_titles = {
        title: hashes
        for title, hashes
        in title_content_hashes.items()
        if len(hashes) > 1
    }

    # Sort conflicting titles by number of versions.
    conflicting_title_summary = sorted(
        (
            (
                title,
                len(hashes),
                title_occurrences[title],
            )
            for title, hashes
            in conflicting_titles.items()
        ),
        key=lambda item: (
            item[1],
            item[2],
        ),
        reverse=True,
    )

    # --------------------------------------------------
    # 5. PRINT CORPUS SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("CORPUS SUMMARY")
    print("=" * 70)

    print(
        f"QA examples: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Total document occurrences: "
        f"{total_document_occurrences:,}"
    )

    print(
        f"Unique normalized titles: "
        f"{len(title_occurrences):,}"
    )

    print(
        f"Repeated titles: "
        f"{len(repeated_titles):,}"
    )

    print(
        f"Titles with multiple content versions: "
        f"{len(conflicting_titles):,}"
    )

    # --------------------------------------------------
    # 6. SENTENCE STRUCTURE
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("SENTENCE STRUCTURE")
    print("=" * 70)

    print(
        f"Documents containing empty sentences: "
        f"{documents_with_empty_sentences:,}"
    )

    print(
        f"Total empty sentence positions: "
        f"{empty_sentence_count:,}"
    )

    # --------------------------------------------------
    # 7. GOLD EVIDENCE
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("GOLD EVIDENCE AUDIT")
    print("=" * 70)

    print(
        f"Total supporting facts: "
        f"{total_supporting_facts:,}"
    )

    print(
        f"Valid supporting facts: "
        f"{valid_supporting_facts:,}"
    )

    print(
        f"Missing supporting documents: "
        f"{missing_supporting_documents:,}"
    )

    print(
        f"Invalid sentence IDs: "
        f"{invalid_sentence_ids:,}"
    )

    print(
        f"Gold facts pointing to empty sentences: "
        f"{empty_gold_sentences:,}"
    )

    print(
        f"Examples containing malformed gold evidence: "
        f"{len(malformed_examples):,}"
    )

    # --------------------------------------------------
    # 8. INVALID EXAMPLES
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("INVALID GOLD EVIDENCE EXAMPLES")
    print("=" * 70)

    if not invalid_examples:
        print(
            "\nNo malformed gold evidence found."
        )

    else:
        print(
            f"\nTotal problems: "
            f"{len(invalid_examples):,}"
        )

        # Print at most the first 20 so the terminal
        # remains readable.
        for problem_number, problem in enumerate(
            invalid_examples[:20],
            start=1,
        ):
            print("\n" + "-" * 70)
            print(
                f"Problem #{problem_number}"
            )

            for key, value in problem.items():
                print(
                    f"{key}: {value}"
                )

        if len(invalid_examples) > 20:
            print(
                f"\n... "
                f"{len(invalid_examples) - 20:,} "
                f"additional problems not printed."
            )

    # --------------------------------------------------
    # 9. CONTENT-CONFLICT EXAMPLES
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("TITLE CONTENT CONFLICTS")
    print("=" * 70)

    if not conflicting_title_summary:
        print(
            "\nNo title/content conflicts found."
        )

    else:
        print(
            f"\nTitles with multiple versions: "
            f"{len(conflicting_title_summary):,}"
        )

        print(
            "\nTop 20 conflicting titles:"
        )

        for (
            title,
            version_count,
            occurrence_count,
        ) in conflicting_title_summary[:20]:

            print("\n" + "-" * 70)

            print(
                f"Title: {title}"
            )

            print(
                f"Occurrences: "
                f"{occurrence_count:,}"
            )

            print(
                f"Distinct content versions: "
                f"{version_count:,}"
            )

            versions = list(
                title_versions[
                    title
                ].values()
            )

            for version_index, sentences in enumerate(
                versions[:3],
                start=1,
            ):
                print(
                    f"\n  Version "
                    f"{version_index}:"
                )

                print(
                    f"  Sentence count: "
                    f"{len(sentences)}"
                )

                preview = " ".join(
                    sentence
                    for sentence in sentences
                    if sentence
                )

                print(
                    f"  Preview: "
                    f"{preview[:300]}"
                )

            if len(versions) > 3:
                print(
                    f"\n  ... "
                    f"{len(versions) - 3:,} "
                    f"additional versions."
                )

    # --------------------------------------------------
    # 10. FINAL
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("FULL HOTPOTQA AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()