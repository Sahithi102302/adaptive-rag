
"""Phase 11.4G-12H-5: ambiguity diagnostics for extractor v4.

These cases probe whether a title can be confused with a
question-level constraint. This is a diagnostic, not a proof
of general boundary safety.
"""

from src.recovery.title_span_extractor import (
    ConservativeTitleSpanExtractor,
    TitleSpanStatus,
)


CASES = [
    (
        "Which university did the author of "
        "The Copper Horizon never attend?",
        "The Copper Horizon",
    ),
    (
        "Which university did the author of "
        "The Copper Horizon attend in 1998?",
        "The Copper Horizon",
    ),
    (
        "Which university did the author of "
        "The Copper Horizon attend first?",
        "The Copper Horizon",
    ),
    (
        "Which university did the author of "
        "The Copper Horizon attend before 2005?",
        "The Copper Horizon",
    ),
    (
        "Which university did the author of "
        "The Attend Before 2005 Handbook attend?",
        "The Attend Before 2005 Handbook",
    ),
    (
        "Which university did the author of "
        "The Never Attend Guide attend?",
        "The Never Attend Guide",
    ),
    (
        "Which university did the author of "
        "The Last Meridian attend?",
        "The Last Meridian",
    ),
    (
        "Which university did the author of "
        "Summer in 1998 attend?",
        "Summer in 1998",
    ),
]


def main() -> None:
    extractor = ConservativeTitleSpanExtractor()
    exact = 0
    wrong = 0
    unresolved = 0

    for index, (question, expected) in enumerate(CASES, 1):
        result = extractor.extract(question)

        actual = (
            result.title
            if result.status == TitleSpanStatus.FOUND
            else None
        )

        if actual == expected:
            outcome = "EXACT"
            exact += 1
        elif actual is None:
            outcome = "UNRESOLVED"
            unresolved += 1
        else:
            outcome = "WRONG_SPAN"
            wrong += 1

        print(f"V{index:02d}: {outcome}")
        print(f"  Expected: {expected!r}")
        print(f"  Actual:   {actual!r}")
        print(f"  Reason:   {result.reason}")

    print("\nPHASE 11.4G-12H-5 AMBIGUITY DIAGNOSTIC")
    print(f"Exact: {exact}/{len(CASES)}")
    print(f"Wrong spans: {wrong}")
    print(f"Unresolved: {unresolved}")


if __name__ == "__main__":
    main()
