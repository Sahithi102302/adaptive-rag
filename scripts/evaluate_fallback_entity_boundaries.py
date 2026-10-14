
"""Phase 11.4G-12G: entity-boundary robustness evaluation.

Evaluates whether the semantic fallback policy incorrectly treats
words inside book titles as question-level semantic constraints.

Also checks whether genuine unsupported constraints are screened.

Important:
- This is a targeted diagnostic set, not independent validation.
- Expected labels are manually assigned.
- CANDIDATE means further review only, never acceptance.
- No Qwen inference is required.
- Do not modify the policy before running this evaluation.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.recovery.semantic_fallback_policy import (
    FallbackEligibility,
    SemanticFallbackPolicy,
)


OUTPUT_PATH = Path(
    "outputs/phase11_4g12g_entity_boundary_results.json"
)


CASES = [
    # Supported: misleading words are part of book titles.
    {
        "id": "B01",
        "category": "title_negation",
        "question": (
            "Which university did the author of "
            "The Never Ending Journey attend?"
        ),
        "expected_supported": True,
    },
    {
        "id": "B02",
        "category": "title_additional_constraint",
        "question": (
            "Which university did the author of "
            "Before the Sunrise attend?"
        ),
        "expected_supported": True,
    },
    {
        "id": "B03",
        "category": "title_role",
        "question": (
            "Which university did the author of "
            "The Editor's Notebook attend?"
        ),
        "expected_supported": True,
    },
    {
        "id": "B04",
        "category": "title_uncertainty",
        "question": (
            "Which university did the author of "
            "Perhaps Tomorrow attend?"
        ),
        "expected_supported": True,
    },
    {
        "id": "B05",
        "category": "title_temporal",
        "question": (
            "Which university did the author of "
            "Summer in 1998 attend?"
        ),
        "expected_supported": True,
    },
    {
        "id": "B06",
        "category": "title_relation",
        "question": (
            "Which university did the author of "
            "The University That Hired Me attend?"
        ),
        "expected_supported": True,
    },
    {
        "id": "B07",
        "category": "title_compound",
        "question": (
            "Which university did the author of "
            "Where and When attend?"
        ),
        "expected_supported": True,
    },
    {
        "id": "B08",
        "category": "title_additional_constraint",
        "question": (
            "Which university did the author of "
            "After the Rain attend?"
        ),
        "expected_supported": True,
    },
    {
        "id": "B09",
        "category": "title_role",
        "question": (
            "Which university did the author of "
            "The Narrator attend?"
        ),
        "expected_supported": True,
    },
    {
        "id": "B10",
        "category": "title_safe_control",
        "question": (
            "Which university did the author of "
            "The Copper Horizon attend?"
        ),
        "expected_supported": True,
    },

    # Unsupported: the extra constraints apply to the question.
    {
        "id": "B11",
        "category": "actual_negation",
        "question": (
            "Which university did the author of "
            "The Copper Horizon never attend?"
        ),
        "expected_supported": False,
    },
    {
        "id": "B12",
        "category": "actual_additional_constraint",
        "question": (
            "Which university did the author of "
            "The Copper Horizon attend before 2005?"
        ),
        "expected_supported": False,
    },
    {
        "id": "B13",
        "category": "actual_role",
        "question": (
            "Which university did the editor of "
            "The Copper Horizon attend?"
        ),
        "expected_supported": False,
    },
    {
        "id": "B14",
        "category": "actual_uncertainty",
        "question": (
            "Which university might the author of "
            "The Copper Horizon have attended?"
        ),
        "expected_supported": False,
    },
    {
        "id": "B15",
        "category": "actual_temporal",
        "question": (
            "Which university did the author of "
            "The Copper Horizon attend in 1998?"
        ),
        "expected_supported": False,
    },
    {
        "id": "B16",
        "category": "actual_relation",
        "question": (
            "Which university employed the author of "
            "The Copper Horizon?"
        ),
        "expected_supported": False,
    },
    {
        "id": "B17",
        "category": "actual_compound",
        "question": (
            "Which university did the author of "
            "The Copper Horizon attend, and when?"
        ),
        "expected_supported": False,
    },
    {
        "id": "B18",
        "category": "actual_additional_constraint",
        "question": (
            "Which university did the author of "
            "The Copper Horizon attend first?"
        ),
        "expected_supported": False,
    },
    {
        "id": "B19",
        "category": "actual_role",
        "question": (
            "Which university did the narrator of "
            "The Copper Horizon attend?"
        ),
        "expected_supported": False,
    },
    {
        "id": "B20",
        "category": "actual_one_hop",
        "question": "Who authored The Copper Horizon?",
        "expected_supported": False,
    },
]


def main() -> None:
    policy = SemanticFallbackPolicy()

    records = []
    counts = {
        "supported": Counter(),
        "unsupported": Counter(),
    }

    print("=" * 72)
    print("PHASE 11.4G-12G — ENTITY-BOUNDARY ROBUSTNESS")
    print(f"Policy: {policy.name}")
    print("=" * 72)

    for case in CASES:
        result = policy.screen(case["question"])

        group = (
            "supported"
            if case["expected_supported"]
            else "unsupported"
        )

        eligibility = result.eligibility.value
        counts[group][eligibility] += 1

        record = {
            **case,
            "eligibility": eligibility,
            "reason": result.reason,
        }
        records.append(record)

        print(
            f"{case['id']} [{case['category']}]: "
            f"{group} -> {eligibility} "
            f"({result.reason})"
        )

    supported_total = sum(
        case["expected_supported"] for case in CASES
    )
    unsupported_total = len(CASES) - supported_total

    supported_candidates = counts["supported"]["candidate"]
    supported_blocked = counts["supported"]["blocked"]
    supported_unresolved = counts["supported"]["unresolved"]

    unsupported_candidates = counts["unsupported"]["candidate"]
    unsupported_blocked = counts["unsupported"]["blocked"]
    unsupported_unresolved = counts["unsupported"]["unresolved"]

    summary = {
        "total": len(CASES),
        "supported_total": supported_total,
        "unsupported_total": unsupported_total,
        "supported_candidates": supported_candidates,
        "supported_blocked": supported_blocked,
        "supported_unresolved": supported_unresolved,
        "unsupported_candidates": unsupported_candidates,
        "unsupported_blocked": unsupported_blocked,
        "unsupported_unresolved": unsupported_unresolved,
        "supported_candidate_coverage": (
            supported_candidates / supported_total
            if supported_total else 0.0
        ),
        "unsafe_candidate_rate": (
            unsupported_candidates / unsupported_total
            if unsupported_total else 0.0
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "experiment": "phase11_4g12g",
                "policy": policy.name,
                "evaluation_type": "targeted_diagnostic",
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

    print("\nSUPPORTED QUESTIONS")
    print(f"  Total: {supported_total}")
    print(f"  Candidates: {supported_candidates}")
    print(f"  Blocked: {supported_blocked}")
    print(f"  Unresolved: {supported_unresolved}")
    print(
        "  Candidate coverage: "
        f"{summary['supported_candidate_coverage']:.1%}"
    )

    print("\nUNSUPPORTED QUESTIONS")
    print(f"  Total: {unsupported_total}")
    print(f"  Unsafe candidates: {unsupported_candidates}")
    print(f"  Blocked: {unsupported_blocked}")
    print(f"  Unresolved: {unsupported_unresolved}")
    print(
        "  Unsafe candidate rate: "
        f"{summary['unsafe_candidate_rate']:.1%}"
    )

    print("\nSUPPORTED QUESTIONS NOT MARKED AS CANDIDATES")
    missed = [
        record for record in records
        if record["expected_supported"]
        and record["eligibility"]
        != FallbackEligibility.CANDIDATE.value
    ]

    if not missed:
        print("  None.")
    else:
        for record in missed:
            print(
                f"  {record['id']}: {record['reason']} | "
                f"{record['question']}"
            )

    print("\nUNSUPPORTED QUESTIONS MARKED AS CANDIDATES")
    unsafe = [
        record for record in records
        if not record["expected_supported"]
        and record["eligibility"]
        == FallbackEligibility.CANDIDATE.value
    ]

    if not unsafe:
        print("  None.")
    else:
        for record in unsafe:
            print(
                f"  {record['id']}: {record['question']}"
            )

    print(f"\nSaved results: {OUTPUT_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
