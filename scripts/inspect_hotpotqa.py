from datasets import load_dataset


def main() -> None:
    print("=" * 70)
    print("ADAPTIVERAG - HOTPOTQA DATASET INSPECTION")
    print("=" * 70)

    print("\nLoading HotpotQA...")
    print("The first run may take some time because the dataset must download.")

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    print("\nDataset loaded successfully.")

    print("\n" + "=" * 70)
    print("DATASET STRUCTURE")
    print("=" * 70)

    print(dataset)

    print("\n" + "=" * 70)
    print("SPLIT SIZES")
    print("=" * 70)

    for split_name, split in dataset.items():
        print(f"{split_name}: {len(split):,} examples")

    train_dataset = dataset["train"]

    print("\n" + "=" * 70)
    print("FEATURES")
    print("=" * 70)

    print(train_dataset.features)

    example = train_dataset[0]

    print("\n" + "=" * 70)
    print("FIRST TRAINING EXAMPLE")
    print("=" * 70)

    print(f"\nID:\n{example['id']}")

    print(f"\nQuestion:\n{example['question']}")

    print(f"\nAnswer:\n{example['answer']}")

    print(f"\nType:\n{example['type']}")

    print(f"\nLevel:\n{example['level']}")

    print("\nCONTEXT DOCUMENTS:")

    context = example["context"]

    titles = context["title"]
    sentence_groups = context["sentences"]

    for document_index, (title, sentences) in enumerate(
        zip(titles, sentence_groups),
        start=1,
    ):
        print("\n" + "-" * 60)
        print(f"Document #{document_index}")
        print(f"Title: {title}")

        for sentence_index, sentence in enumerate(sentences):
            print(f"  [{sentence_index}] {sentence}")

    print("\n" + "=" * 70)
    print("SUPPORTING FACTS")
    print("=" * 70)

    supporting_facts = example["supporting_facts"]

    supporting_titles = supporting_facts["title"]
    supporting_sentence_ids = supporting_facts["sent_id"]

    for title, sentence_id in zip(
        supporting_titles,
        supporting_sentence_ids,
    ):
        print(
            f"Document: {title} | "
            f"Sentence ID: {sentence_id}"
        )

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()