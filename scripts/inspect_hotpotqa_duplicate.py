from datasets import load_dataset


TARGET_TITLE = "Minoru Suzuki"
TARGET_EXAMPLE_ID = "5a7b23ca554299042af8f703"
DEVELOPMENT_EXAMPLES = 1000


def main() -> None:
    print("=" * 70)
    print("HOTPOTQA DOCUMENT DUPLICATE INVESTIGATION")
    print("=" * 70)

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    train_dataset = dataset["train"]

    subset = train_dataset.select(
        range(
            min(
                DEVELOPMENT_EXAMPLES,
                len(train_dataset),
            )
        )
    )

    occurrences = []

    for example_index, record in enumerate(subset):

        titles = record["context"]["title"]
        sentence_groups = record["context"]["sentences"]

        for title, sentences in zip(
            titles,
            sentence_groups,
        ):

            if title.strip() == TARGET_TITLE:

                occurrences.append(
                    {
                        "example_index": example_index,
                        "example_id": record["id"],
                        "question": record["question"],
                        "sentences": sentences,
                        "supporting_titles":
                            record["supporting_facts"]["title"],
                        "supporting_sentence_ids":
                            record["supporting_facts"]["sent_id"],
                    }
                )

    print(
        f"\nOccurrences of '{TARGET_TITLE}': "
        f"{len(occurrences)}"
    )

    for occurrence_index, occurrence in enumerate(
        occurrences,
        start=1,
    ):
        print("\n" + "=" * 70)
        print(f"OCCURRENCE #{occurrence_index}")
        print("=" * 70)

        print(
            f"Dataset index: "
            f"{occurrence['example_index']}"
        )

        print(
            f"Example ID: "
            f"{occurrence['example_id']}"
        )

        print(
            f"Target problem example: "
            f"{occurrence['example_id'] == TARGET_EXAMPLE_ID}"
        )

        print(
            f"Question: "
            f"{occurrence['question']}"
        )

        sentences = occurrence["sentences"]

        print(
            f"Sentence count: "
            f"{len(sentences)}"
        )

        print("\nSentences:")

        for sentence_index, sentence in enumerate(
            sentences
        ):
            print(
                f"  [{sentence_index}] "
                f"{repr(sentence)}"
            )

        print("\nSupporting facts for this question:")

        for title, sentence_id in zip(
            occurrence["supporting_titles"],
            occurrence["supporting_sentence_ids"],
        ):
            print(
                f"  {title} | "
                f"Sentence ID: {sentence_id}"
            )


if __name__ == "__main__":
    main()