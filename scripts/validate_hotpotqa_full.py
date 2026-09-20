from datasets import load_dataset

from src.ingestion.hotpotqa import (
    build_hotpotqa_corpus,
    validate_gold_evidence,
    find_invalid_gold_evidence,
)


def main() -> None:
    print("=" * 70)
    print("ADAPTIVERAG - FULL HOTPOTQA ADAPTER VALIDATION")
    print("=" * 70)

    print("\nLoading HotpotQA...")

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    train_dataset = dataset["train"]

    print(
        f"Training examples: "
        f"{len(train_dataset):,}"
    )

    print(
        "\nBuilding full version-aware retrieval corpus..."
    )

    corpus = build_hotpotqa_corpus(
        train_dataset
    )

    print("\n" + "=" * 70)
    print("CORPUS")
    print("=" * 70)

    print(
        f"QA examples: "
        f"{len(corpus.examples):,}"
    )

    print(
        f"Unique document versions: "
        f"{len(corpus.documents):,}"
    )

    print("\n" + "=" * 70)
    print("GOLD EVIDENCE")
    print("=" * 70)

    validation = validate_gold_evidence(
        corpus
    )

    for key, value in validation.items():
        print(
            f"{key}: {value:,}"
        )

    problems = find_invalid_gold_evidence(
        corpus
    )

    print("\n" + "=" * 70)
    print("PROBLEM SUMMARY")
    print("=" * 70)

    print(
        f"Total problems: "
        f"{len(problems):,}"
    )

    problem_types = {}

    for problem in problems:
        problem_type = problem["problem"]

        problem_types[problem_type] = (
            problem_types.get(
                problem_type,
                0,
            )
            + 1
        )

    for problem_type, count in sorted(
        problem_types.items()
    ):
        print(
            f"{problem_type}: "
            f"{count:,}"
        )

    print("\nFirst 10 problems:")

    for index, problem in enumerate(
        problems[:10],
        start=1,
    ):
        print("\n" + "-" * 70)

        print(
            f"Problem #{index}"
        )

        for key, value in problem.items():
            print(
                f"{key}: {value}"
            )

    print("\n" + "=" * 70)
    print("FULL ADAPTER VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()