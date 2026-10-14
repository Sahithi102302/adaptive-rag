
"""Phase 11.4G-12H-11F: adversarial lexical-preservation tests.

These tests intentionally demonstrate that lexical presence
does not establish semantic preservation.

No semantic acceptance is authorized.
"""

from __future__ import annotations

from src.recovery.constraint_preservation_audit import (
    ConstraintPreservationAuditor,
    LexicalStatus,
)


CASES = (
    (
        "A01_reversed_temporal",
        "Who held the position before 1998?",
        (
            "Who held the position after 1998?",
            "What happened before that?",
        ),
        (LexicalStatus.PRESENT, LexicalStatus.PRESENT),
        "Lexical presence can coexist with reversed temporal meaning.",
    ),
    (
        "A02_unrelated_copy",
        "Who never attended Harvard University?",
        (
            "Who attended Harvard University?",
            "What does the word never mean?",
        ),
        (LexicalStatus.PRESENT,),
        "Copying a constraint into an unrelated subquestion is insufficient.",
    ),
    (
        "A03_negation_removed",
        "Who never attended Harvard University?",
        (
            "Who attended Harvard University?",
        ),
        (LexicalStatus.MISSING,),
        "The original negation token is absent.",
    ),
    (
        "A04_ambiguous_copy",
        "Which team does Lewis Young's older brother play for?",
        (
            "Which team does Lewis Young's older brother play for?",
        ),
        (LexicalStatus.UNRESOLVED,),
        "Ambiguity remains unresolved despite exact copying.",
    ),
    (
        "A05_semantic_paraphrase",
        "Who held the position before 1998?",
        (
            "Who held the position prior to 1998?",
        ),
        (LexicalStatus.MISSING, LexicalStatus.PRESENT),
        "A valid paraphrase may lack the original lexical signal.",
    ),
)


def main() -> None:
    auditor = ConstraintPreservationAuditor()
    passed = 0

    for (
        case_id,
        question,
        subquestions,
        expected,
        explanation,
    ) in CASES:
        result = auditor.audit(question, subquestions)

        actual = tuple(
            item.status for item in result.observations
        )

        assert actual == expected, (
            f"{case_id}: expected {expected}, got {actual}"
        )

        passed += 1

        print(f"\nPASS {case_id}")
        print(f"  statuses={[status.value for status in actual]}")
        print(f"  limitation={explanation}")

    print(
        f"\nAdversarial preservation tests: "
        f"{passed}/{len(CASES)}"
    )


if __name__ == "__main__":
    main()
