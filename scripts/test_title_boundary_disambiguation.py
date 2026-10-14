
"""Phase 11.4G-12H-9: evidence-guided disambiguation tests."""

from src.recovery.title_boundary_validator import (
    BoundaryStatus,
    EvidenceBackedTitleBoundaryValidator,
)


CASES = [
    (
        "D01",
        "Which university did the author of The Copper Horizon attend?",
        ["The Copper Horizon"],
        "The Copper Horizon",
    ),
    (
        "D02",
        "Which university did the author of The Copper Horizon never attend?",
        ["The Copper Horizon"],
        "The Copper Horizon",
    ),
    (
        "D03",
        "Which university did the author of The Attend Before 2005 Handbook attend?",
        ["The Attend Before 2005 Handbook"],
        "The Attend Before 2005 Handbook",
    ),
    (
        "D04",
        "Which university did the author of The Attend Before 2005 Handbook attend?",
        ["The", "The Attend Before 2005 Handbook"],
        None,
    ),
    (
        "D05",
        "Which university did the author of The Copper Horizon attend?",
        ["An Unrelated Book"],
        None,
    ),
    (
        "D06",
        "At what college was the author of The Last Meridian educated?",
        ["The Last Meridian"],
        "The Last Meridian",
    ),
    (
        "D07",
        "At what college was the author of The Last Meridian attend?",
        ["The Last Meridian"],
        None,
    ),
    (
        "D08",
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
        f"\nEvidence-guided disambiguation: "
        f"{passed}/{len(CASES)}"
    )

    if passed != len(CASES):
        raise AssertionError(
            "Evidence-guided disambiguation failed."
        )


if __name__ == "__main__":
    main()
