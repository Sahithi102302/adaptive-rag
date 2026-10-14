
"""Tests for Phase 11.4G-12H-11C signal review."""

from __future__ import annotations

from src.recovery.question_constraint_inventory import (
    QuestionConstraintInventory,
)
from src.recovery.constraint_signal_review import (
    ConstraintSignalReviewer,
    SignalReviewStatus,
)


CASES = (
    (
        "R01",
        "Which show aired from September 22, 1994, to May 6, 2004?",
        "may",
        SignalReviewStatus.POTENTIALLY_AMBIGUOUS,
        "possible_calendar_month",
    ),
    (
        "R02",
        "What country does Eccles, Greater Manchester belong to?",
        "greater",
        SignalReviewStatus.POTENTIALLY_AMBIGUOUS,
        "possible_geographic_name",
    ),
    (
        "R03",
        "Which team does Lewis Young's older brother play for?",
        "older",
        SignalReviewStatus.POTENTIALLY_AMBIGUOUS,
        "possible_family_relation_modifier",
    ),
    (
        "R04",
        "Who is older, Person A or Person B?",
        "older",
        SignalReviewStatus.NEEDS_REVIEW,
        "surface_signal_requires_interpretation",
    ),
    (
        "R05",
        "Who may have written the book?",
        "may",
        SignalReviewStatus.NEEDS_REVIEW,
        "surface_signal_requires_interpretation",
    ),
)


def main() -> None:
    inventory = QuestionConstraintInventory()
    reviewer = ConstraintSignalReviewer()

    passed = 0

    for case_id, question, target, expected_status, expected_reason in CASES:
        detected = inventory.analyze(question)
        reviewed = reviewer.review(detected)

        assert len(reviewed.reviewed_signals) == len(detected.signals)

        matches = [
            item
            for item in reviewed.reviewed_signals
            if item.signal.text.lower() == target
        ]

        assert len(matches) == 1, (
            f"{case_id}: expected one {target!r} signal"
        )

        item = matches[0]

        assert item.status == expected_status, (
            f"{case_id}: unexpected status {item.status}"
        )

        assert item.reason == expected_reason, (
            f"{case_id}: unexpected reason {item.reason}"
        )

        passed += 1

        print(
            f"PASS {case_id}: "
            f"{item.signal.text!r} -> "
            f"{item.status.value} ({item.reason})"
        )

    print(f"\nSignal review tests: {passed}/{len(CASES)}")


if __name__ == "__main__":
    main()
