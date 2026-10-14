
"""Phase 11.4G-3: question decomposition baseline evaluation.

Measures template coverage, decomposition correctness, and
abstention behavior on a small, hand-authored diagnostic set.

No retrieved evidence or gold HotpotQA facts are used at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.recovery.question_decomposer import (
    ConservativeQuestionDecomposer,
)


class FailureCategory(str, Enum):
    NONE = "none"
    UNSUPPORTED_TEMPLATE = "unsupported_template"
    INCORRECT_ACCEPTANCE = "incorrect_acceptance"
    WRONG_ENTITY = "wrong_entity"
    WRONG_RELATIONSHIP = "wrong_relationship"
    WRONG_ANSWER_TARGET = "wrong_answer_target"


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    expected_supported: bool
    expected_title: str | None = None
    category_if_rejected: FailureCategory = (
        FailureCategory.UNSUPPORTED_TEMPLATE
    )


CASES = (
    EvaluationCase(
        "Q01",
        "Which university did the author of The Silver Atlas attend?",
        True,
        "The Silver Atlas",
    ),
    EvaluationCase(
        "Q02",
        'Which university did the author of "The Silver Atlas" attend?',
        True,
        "The Silver Atlas",
    ),
    EvaluationCase(
        "Q03",
        "which university did the author of The Silver Atlas attend?",
        True,
        "The Silver Atlas",
    ),
    EvaluationCase(
        "Q04",
        "Which university did the author of The Hidden Garden attend?",
        True,
        "The Hidden Garden",
    ),
    EvaluationCase(
        "Q05",
        "Where did the writer of The Silver Atlas study?",
        True,
        "The Silver Atlas",
    ),
    EvaluationCase(
        "Q06",
        "What university did the author of The Silver Atlas attend?",
        True,
        "The Silver Atlas",
    ),
    EvaluationCase(
        "Q07",
        "Which college did the author of The Silver Atlas attend?",
        True,
        "The Silver Atlas",
    ),
    EvaluationCase(
        "Q08",
        "At which university did the writer of The Silver Atlas study?",
        True,
        "The Silver Atlas",
    ),
    EvaluationCase(
        "Q09",
        "Who wrote The Silver Atlas?",
        False,
    ),
    EvaluationCase(
        "Q10",
        "Which city is the author's university located in?",
        False,
    ),
    EvaluationCase(
        "Q11",
        "Which university did the author of The Silver Atlas teach at?",
        False,
    ),
    EvaluationCase(
        "Q12",
        "When was The Silver Atlas published?",
        False,
    ),
    EvaluationCase(
        "Q13",
        "Which university did the illustrator of The Silver Atlas attend?",
        False,
    ),
    EvaluationCase(
        "Q14",
        "Which university did the editor of The Silver Atlas attend?",
        False,
    ),
    EvaluationCase(
        "Q15",
        "Which university did the author of The Silver Atlas attend",
        True,
        "The Silver Atlas",
    ),
    EvaluationCase(
        "Q16",
        "Which university did the author of The Silver Atlas attend, "
        "and when did they graduate?",
        False,
    ),
)


def check_decomposition(result, case):
    """Check declared structure, not merely the support flag."""

    if not result.is_usable:
        return False, FailureCategory.UNSUPPORTED_TEMPLATE

    requirements = result.requirements

    if requirements is None:
        return False, FailureCategory.INCORRECT_ACCEPTANCE

    relationships = requirements.relationships

    if len(relationships) != 2:
        return False, FailureCategory.WRONG_RELATIONSHIP

    first, second = relationships

    if first.subject != case.expected_title:
        return False, FailureCategory.WRONG_ENTITY

    if (
        first.predicate != "authored_by"
        or second.predicate != "attended"
        or first.object != "?author"
        or second.subject != "?author"
    ):
        return False, FailureCategory.WRONG_RELATIONSHIP

    if (
        second.object != "?university"
        or requirements.answer_variable != "?university"
    ):
        return False, FailureCategory.WRONG_ANSWER_TARGET

    return True, FailureCategory.NONE


def main():
    decomposer = ConservativeQuestionDecomposer()

    supported_total = 0
    unsupported_total = 0
    correct_supported = 0
    correct_abstentions = 0
    false_acceptances = 0
    missed_supported = 0
    incorrect_decompositions = 0

    for case in CASES:
        result = decomposer.decompose(case.question)

        if case.expected_supported:
            supported_total += 1

            if not result.is_usable:
                missed_supported += 1
                outcome = "MISSED_SUPPORTED"
                category = case.category_if_rejected
            else:
                valid, category = check_decomposition(
                    result, case
                )
                if valid:
                    correct_supported += 1
                    outcome = "CORRECT"
                else:
                    incorrect_decompositions += 1
                    outcome = "WRONG_DECOMPOSITION"

        else:
            unsupported_total += 1

            if result.is_usable:
                false_acceptances += 1
                outcome = "FALSE_ACCEPTANCE"
                category = FailureCategory.INCORRECT_ACCEPTANCE
            else:
                correct_abstentions += 1
                outcome = "CORRECT_ABSTENTION"
                category = FailureCategory.NONE

        print("-" * 65)
        print(f"{case.case_id}: {case.question}")
        print(f"Expected supported: {case.expected_supported}")
        print(f"Observed supported: {result.is_usable}")
        print(f"Outcome: {outcome}")
        print(f"Failure category: {category.value}")

    total = len(CASES)
    correct_total = correct_supported + correct_abstentions

    coverage = (
        correct_supported / supported_total
        if supported_total else 0.0
    )

    abstention_specificity = (
        correct_abstentions / unsupported_total
        if unsupported_total else 0.0
    )

    print("\n" + "=" * 65)
    print("PHASE 11.4G-3 — DECOMPOSITION BASELINE EVALUATION")
    print(f"Cases: {total}")
    print(f"Expected supported: {supported_total}")
    print(f"Expected unsupported: {unsupported_total}")
    print(f"Correct supported decompositions: {correct_supported}")
    print(f"Correct abstentions: {correct_abstentions}")
    print(f"Missed supported questions: {missed_supported}")
    print(f"Incorrect decompositions: {incorrect_decompositions}")
    print(f"False acceptances: {false_acceptances}")
    print(f"Overall correct decisions: {correct_total}/{total}")
    print(f"Supported-question coverage: {coverage:.1%}")
    print(
        "Unsupported-question abstention specificity: "
        f"{abstention_specificity:.1%}"
    )
    print("=" * 65)


if __name__ == "__main__":
    main()
