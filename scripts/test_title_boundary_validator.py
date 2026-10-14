
"""Phase 11.4G-12H-8: title boundary validator tests."""

from src.recovery.title_boundary_validator import (
    BoundaryStatus,
    EvidenceBackedTitleBoundaryValidator,
)


CASES = [
    (
        "E01",
        "Which university did the author of The Copper Horizon attend?",
        ["The Copper Horizon", "The Paper Constellation"],
        "The Copper Horizon",
    ),
    (
        "E02",
        "Which university did the author of The Copper Horizon attend?",
        ["The Paper Constellation"],
        None,
    ),
    (
        "E03",
        "Which university did the author of The Copper Horizon attend?",
        [],
        None,
    ),
    (
        "E04",
        "Which university did the author of Summer in 1998 attend?",
        ["Summer in 1998"],
        "Summer in 1998",
    ),
    (
        "E05",
        "Which university did the author of The Never Attend Guide attend?",
        ["The Never Attend Guide"],
        "The Never Attend Guide",
    ),
    (
    "E06",
    "Which university did the author of The Copper Horizon never attend?",
    ["The Copper Horizon"],
    "The Copper Horizon",
    ),
    (
    "E07",
    "Which university did the author of The Attend Before 2005 Handbook attend?",
    ["The Attend Before 2005 Handbook"],
    "The Attend Before 2005 Handbook",
    ),
    (
        "E08",
        "At what college was the author of The Last Meridian educated?",
        ["The Last Meridian"],
        "The Last Meridian",
    ),
    (
        "E09",
        "Which university did the author of The Copper Horizon attend?",
        ["  the   copper horizon  "],
        "The Copper Horizon",
    ),
    (
        "E10",
        "Who authored The Copper Horizon?",
        ["The Copper Horizon"],
        None,
    ),
]


def main() -> None:
    validator = EvidenceBackedTitleBoundaryValidator()
    passed = 0

    for case_id, question, titles, expected in CASES:
        result = validator.validate(question, titles)

        actual = (
            result.title
            if result.status == BoundaryStatus.VERIFIED_TITLE
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
        f"\nEvidence-backed boundary tests: "
        f"{passed}/{len(CASES)}"
    )

    if passed != len(CASES):
        raise AssertionError(
            "Evidence-backed boundary validation failed."
        )


if __name__ == "__main__":
    main()
