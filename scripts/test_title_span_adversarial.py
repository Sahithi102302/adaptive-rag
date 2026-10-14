
"""Phase 11.4G-12H-2: adversarial title-boundary diagnostics.

Tests the existing extractor without modifying it.

The cases intentionally include ambiguous titles and misleading
boundary words. A wrong extraction is reported, not silently
treated as a safe title span.

This is a targeted development diagnostic, not held-out validation.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.recovery.title_span_extractor import (
    ConservativeTitleSpanExtractor,
    TitleSpanStatus,
)


OUTPUT_PATH = Path(
    "outputs/phase11_4g12h2_adversarial_boundaries.json"
)

CASES = [
    {
        "id": "T01",
        "category": "control",
        "question": "Which university did the author of The Copper Horizon attend?",
        "expected_title": "The Copper Horizon",
    },
    {
        "id": "T02",
        "category": "title_contains_attend",
        "question": "Which university did the author of The Attend Study Handbook attend?",
        "expected_title": "The Attend Study Handbook",
    },
    {
        "id": "T03",
        "category": "title_contains_study",
        "question": "Which university did the author of The Study of Stars attend?",
        "expected_title": "The Study of Stars",
    },
    {
        "id": "T04",
        "category": "title_contains_educated",
        "question": "Which university did the author of The Educated Traveler attend?",
        "expected_title": "The Educated Traveler",
    },
    {
        "id": "T05",
        "category": "title_contains_attended",
        "question": "Which university did the author of The Attended Conference attend?",
        "expected_title": "The Attended Conference",
    },
    {
        "id": "T06",
        "category": "title_contains_go",
        "question": "Which university did the author of The Go Between attend?",
        "expected_title": "The Go Between",
    },
    {
        "id": "T07",
        "category": "title_contains_went",
        "question": "Which university did the author of Where We Went attend?",
        "expected_title": "Where We Went",
    },
    {
        "id": "T08",
        "category": "temporal_constraint",
        "question": "Which university did the author of The Copper Horizon attend in 1998?",
        "expected_title": "The Copper Horizon",
    },
    {
        "id": "T09",
        "category": "ordering_constraint",
        "question": "Which university did the author of The Copper Horizon attend first?",
        "expected_title": "The Copper Horizon",
    },
    {
    "id": "T10",
    "category": "negation_constraint",
    "question": (
        "Which university did the author of "
        "The Copper Horizon never attend?"
    ),
    "expected_title": "The Copper Horizon",
    },
    {
        "id": "T11",
        "category": "ambiguous_missing_continuation",
        "question": "Which university did the author of The Copper Horizon?",
        "expected_title": None,
    },
    {
        "id": "T12",
        "category": "one_hop",
        "question": "Who authored The Copper Horizon?",
        "expected_title": None,
    },
    {
        "id": "T13",
        "category": "alternate_grammar",
        "question": "At what college was the author of The Last Meridian educated?",
        "expected_title": "The Last Meridian",
    },
    {
        "id": "T14",
        "category": "alternate_authorship",
        "question": "Where did the person who authored The Paper Constellation go to college?",
        "expected_title": "The Paper Constellation",
    },
    {
        "id": "T15",
        "category": "title_contains_temporal_expression",
        "question": "Which university did the author of Summer in 1998 attend?",
        "expected_title": "Summer in 1998",
    },
]



def main() -> None:
    extractor = ConservativeTitleSpanExtractor()

    records = []
    exact_matches = 0
    wrong_spans = 0
    unresolved_expected_titles = 0
    unexpected_spans = 0
    correct_abstentions = 0

    print("=" * 72)
    print("PHASE 11.4G-12H-3 — BOUNDARY SAFETY DIAGNOSTICS")
    print(f"Extractor: {extractor.name}")
    print("=" * 72)

    for case in CASES:
        result = extractor.extract(case["question"])
        expected = case["expected_title"]

        actual = (
            result.title
            if result.status == TitleSpanStatus.FOUND
            else None
        )

        if actual == expected:
            if expected is None:
                correct_abstentions += 1
                outcome = "CORRECT_ABSTENTION"
            else:
                exact_matches += 1
                outcome = "EXACT_MATCH"

        elif actual is None:
            unresolved_expected_titles += 1
            outcome = "SAFE_UNRESOLVED"

        elif expected is None:
            unexpected_spans += 1
            outcome = "UNEXPECTED_SPAN"

        else:
            wrong_spans += 1
            outcome = "WRONG_SPAN"

        records.append(
            {
                **case,
                "actual_title": actual,
                "extractor_status": result.status.value,
                "extractor_reason": result.reason,
                "outcome": outcome,
                "start": result.start,
                "end": result.end,
            }
        )

        print(f"\n{case['id']} [{case['category']}]")
        print(f"  Outcome:  {outcome}")
        print(f"  Expected: {expected!r}")
        print(f"  Actual:   {actual!r}")
        print(f"  Reason:   {result.reason}")

    summary = {
        "total": len(CASES),
        "exact_matches": exact_matches,
        "correct_abstentions": correct_abstentions,
        "wrong_spans": wrong_spans,
        "unresolved_expected_titles": unresolved_expected_titles,
        "unexpected_spans": unexpected_spans,
        "unsafe_extractions": wrong_spans + unexpected_spans,
    }

    OUTPUT_PATH = Path(
        "outputs/phase11_4g12h3_boundary_safety_results.json"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "experiment": "phase11_4g12h3",
                "extractor": extractor.name,
                "summary": summary,
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Total cases: {len(CASES)}")
    print(f"Exact matches: {exact_matches}")
    print(f"Correct abstentions: {correct_abstentions}")
    print(f"Wrong spans: {wrong_spans}")
    print(f"Expected titles unresolved: {unresolved_expected_titles}")
    print(f"Unexpected spans: {unexpected_spans}")
    print(
        "Unsafe extractions: "
        f"{summary['unsafe_extractions']}"
    )
    print(f"\nSaved results: {OUTPUT_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
