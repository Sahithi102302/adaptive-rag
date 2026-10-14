
"""Tests for Phase 11.4G-12H-11E lexical preservation audit."""

from __future__ import annotations

from src.recovery.constraint_preservation_audit import (
    ConstraintPreservationAuditor,
    LexicalStatus,
)


CASES = (
    (
        "P01",
        "Who held the position before 1998?",
        ("Who held the position before 1998?",),
        (LexicalStatus.PRESENT, LexicalStatus.PRESENT),
    ),
    (
        "P02",
        "Who held the position before 1998?",
        ("Who held the position?",),
        (LexicalStatus.MISSING, LexicalStatus.MISSING),
    ),
    (
        "P03",
        "Who may have written the book?",
        ("Who wrote the book?",),
        (LexicalStatus.MISSING,),
    ),
    (
        "P04",
        "Which team does Lewis Young's older brother play for?",
        ("Which team does Ashley Young play for?",),
        (LexicalStatus.UNRESOLVED,),
    ),
    (
        "P05",
        "Who founded the organization?",
        ("Who founded the organization?",),
        (),
    ),
)


def main() -> None:
    auditor = ConstraintPreservationAuditor()
    passed = 0

    for case_id, question, subquestions, expected in CASES:
        result = auditor.audit(question, subquestions)

        actual = tuple(
            item.status for item in result.observations
        )

        assert actual == expected, (
            f"{case_id}: expected {expected}, got {actual}"
        )

        passed += 1

        print(
            f"PASS {case_id}: "
            f"statuses={[status.value for status in actual]}"
        )

    print(
        f"\nConstraint preservation audit tests: "
        f"{passed}/{len(CASES)}"
    )


if __name__ == "__main__":
    main()
