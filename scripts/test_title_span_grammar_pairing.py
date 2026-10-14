
"""Phase 11.4G-12H-7: grammar-pairing diagnostics."""

from src.recovery.title_span_extractor import (
    ConservativeTitleSpanExtractor,
    TitleSpanStatus,
)


CASES = [
    (
        "G01",
        "Which university did the author of The Copper Horizon attend?",
        "The Copper Horizon",
    ),
    (
        "G02",
        "At what college was the author of The Copper Horizon educated?",
        "The Copper Horizon",
    ),
    (
        "G03",
        "Where did the person who authored The Paper Constellation go to college?",
        "The Paper Constellation",
    ),
    (
        "G04",
        "At what college was the author of The Copper Horizon attend?",
        None,
    ),
    (
        "G05",
        "Which university did the author of The Copper Horizon educated?",
        None,
    ),
    (
        "G06",
        "Where did the person who authored The Paper Constellation attend?",
        None,
    ),
    (
        "G07",
        "Which university did the author of The Attend Before 2005 Handbook attend?",
        None,
    ),
    (
        "G08",
        "Which university did the author of The Copper Horizon never attend?",
        None,
    ),
    (
        "G09",
        "Which university did the author of Summer in 1998 attend?",
        "Summer in 1998",
    ),
    (
        "G10",
        "Which university did the author of The Never Attend Guide attend?",
        "The Never Attend Guide",
    ),
]


def main() -> None:
    extractor = ConservativeTitleSpanExtractor()
    passed = 0

    for case_id, question, expected in CASES:
        result = extractor.extract(question)

        actual = (
            result.title
            if result.status == TitleSpanStatus.FOUND
            else None
        )

        ok = actual == expected
        passed += int(ok)

        print(
            f"{'PASS' if ok else 'FAIL'} {case_id}: "
            f"expected={expected!r}, actual={actual!r}, "
            f"reason={result.reason}"
        )

    print(
        f"\nGrammar-pairing diagnostics: "
        f"{passed}/{len(CASES)}"
    )

    if passed != len(CASES):
        raise AssertionError(
            "Grammar-pairing diagnostics failed."
        )


if __name__ == "__main__":
    main()
