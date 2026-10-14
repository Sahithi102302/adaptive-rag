
"""Tests for Phase 11.4G-12H-11A constraint inventory."""

from __future__ import annotations

from src.recovery.question_constraint_inventory import (
    ConstraintKind,
    QuestionConstraintInventory,
)


CASES = (
    (
        "C01",
        "Which university did the author not attend?",
        {ConstraintKind.NEGATION},
    ),
    (
        "C02",
        "Who held the position before 1998?",
        {ConstraintKind.TEMPORAL},
    ),
    (
        "C03",
        "Who was older, Person A or Person B?",
        {ConstraintKind.COMPARATIVE},
    ),
    (
        "C04",
        "Who was the first director?",
        {ConstraintKind.SUPERLATIVE},
    ),
    (
        "C05",
        "Who might have written the book?",
        {ConstraintKind.UNCERTAINTY},
    ),
    (
        "C06",
        "Who portrayed Corliss Archer in the film Kiss and Tell?",
        {ConstraintKind.CONTEXTUAL},
    ),
    (
        "C07",
        "Who founded the organization?",
        set(),
    ),
    (
        "C08",
        "",
        set(),
    ),
)


def main() -> None:
    inventory = QuestionConstraintInventory()

    passed = 0

    for case_id, question, expected in CASES:
        result = inventory.analyze(question)

        actual = {signal.kind for signal in result.signals}

        assert actual == expected, (
            f"{case_id}: expected {expected}, got {actual}"
        )

        for signal in result.signals:
            assert (
                question[signal.start:signal.end] == signal.text
            ), f"{case_id}: invalid character offsets"

        passed += 1
        print(
            f"PASS {case_id}: "
            f"signals={[signal.kind.value for signal in result.signals]}"
        )

    print(f"\nConstraint inventory tests: {passed}/{len(CASES)}")


if __name__ == "__main__":
    main()
