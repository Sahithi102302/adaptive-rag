
"""Phase 11.4G-5: Qwen decomposition smoke test."""

from src.recovery.qwen_question_decomposer import (
    QwenQuestionDecomposer,
)


QUESTIONS = (
    (
        "EXACT_TEMPLATE",
        "Which university did the author of "
        "The Silver Atlas attend?",
    ),
    (
        "PARAPHRASE",
        "Where did the writer of The Silver Atlas study?",
    ),
    (
        "UNSUPPORTED",
        "When was The Silver Atlas published?",
    ),
)


def main():
    print("Loading Qwen question decomposer...")

    decomposer = QwenQuestionDecomposer()

    for case_id, question in QUESTIONS:
        print("\n" + "=" * 65)
        print(f"CASE: {case_id}")
        print(f"Question: {question}")

        result = decomposer.decompose(question)

        print(f"Status: {result.status.value}")
        print(f"Raw response:\n{result.raw_response}")
        print(f"Explanation: {result.explanation}")

        if result.validation is not None:
            print(
                "Structural validation: "
                f"{result.validation.status.value}"
            )

            if result.validation.requirements is not None:
                requirements = result.validation.requirements

                print(
                    "Answer variable: "
                    f"{requirements.answer_variable}"
                )

                for relation in requirements.relationships:
                    print(
                        f"  {relation.relationship_id}: "
                        f"{relation.subject} --"
                        f"{relation.predicate}--> "
                        f"{relation.object}"
                    )

    print("\n" + "=" * 65)
    print("PHASE 11.4G-5 QWEN DECOMPOSITION SMOKE TEST FINISHED")
    print("Review semantic correctness manually.")
    print("=" * 65)


if __name__ == "__main__":
    main()
