
"""Phase 11.4G-12E: semantic fallback safety evaluation.

Evaluates the heuristic fallback screening policy on the
saved abstentions from Phase 11.4G-12C.

This is a development-set diagnostic, not held-out validation.

CANDIDATE does not authorize acceptance or answer generation.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from src.recovery.semantic_fallback_policy import (
    FallbackEligibility,
    SemanticFallbackPolicy,
)


INPUT_PATH = Path(
    "outputs/phase11_4g12_hybrid_router_results.json"
)

OUTPUT_PATH = Path(
    "outputs/phase11_4g12e_fallback_safety_results.json"
)


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing Phase 11.4G-12C results: {INPUT_PATH}"
        )

    payload = json.loads(
        INPUT_PATH.read_text(encoding="utf-8")
    )

    all_records = payload["records"]

    abstentions = [
        record
        for record in all_records
        if record["decision"] == "abstain"
    ]

    if not abstentions:
        raise ValueError("No abstentions found in saved results.")

    policy = SemanticFallbackPolicy()

    output_records = []
    counts = defaultdict(Counter)
    category_counts = defaultdict(Counter)

    for record in abstentions:
        result = policy.screen(record["question"])

        expected_supported = record["expected_supported"]

        group = (
            "supported"
            if expected_supported
            else "unsupported"
        )

        eligibility = result.eligibility.value

        counts[group][eligibility] += 1
        category_counts[record["category"]][eligibility] += 1

        output_records.append(
            {
                "case_id": record["case_id"],
                "category": record["category"],
                "question": record["question"],
                "expected_supported": expected_supported,
                "router_decision": record["decision"],
                "fallback_eligibility": eligibility,
                "fallback_reason": result.reason,
            }
        )

        print(
            f"{record['case_id']} "
            f"[{record['category']}]: "
            f"{group} -> {eligibility} "
            f"({result.reason})"
        )

    supported_total = sum(
        record["expected_supported"]
        for record in abstentions
    )

    unsupported_total = (
        len(abstentions) - supported_total
    )

    supported_candidates = counts["supported"]["candidate"]
    unsupported_candidates = counts["unsupported"]["candidate"]

    supported_blocked = counts["supported"]["blocked"]
    unsupported_blocked = counts["unsupported"]["blocked"]

    supported_unresolved = counts["supported"]["unresolved"]
    unsupported_unresolved = counts["unsupported"]["unresolved"]

    summary = {
        "total_abstentions": len(abstentions),
        "supported_abstentions": supported_total,
        "unsupported_abstentions": unsupported_total,
        "supported_candidates": supported_candidates,
        "supported_blocked": supported_blocked,
        "supported_unresolved": supported_unresolved,
        "unsupported_candidates": unsupported_candidates,
        "unsupported_blocked": unsupported_blocked,
        "unsupported_unresolved": unsupported_unresolved,
        "supported_candidate_coverage": safe_rate(
            supported_candidates,
            supported_total,
        ),
        "unsafe_candidate_rate": safe_rate(
            unsupported_candidates,
            unsupported_total,
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "experiment": "phase11_4g12e",
                "source": str(INPUT_PATH),
                "summary": summary,
                "records": output_records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 68)
    print("PHASE 11.4G-12E — FALLBACK SAFETY RESULTS")
    print(f"Total abstentions: {len(abstentions)}")

    print("\nSUPPORTED ABSTENTIONS")
    print(f"  Total: {supported_total}")
    print(f"  Candidates: {supported_candidates}")
    print(f"  Blocked: {supported_blocked}")
    print(f"  Unresolved: {supported_unresolved}")
    print(
        "  Candidate coverage: "
        f"{summary['supported_candidate_coverage']:.1%}"
    )

    print("\nUNSUPPORTED ABSTENTIONS")
    print(f"  Total: {unsupported_total}")
    print(f"  Unsafe candidates: {unsupported_candidates}")
    print(f"  Blocked: {unsupported_blocked}")
    print(f"  Unresolved: {unsupported_unresolved}")
    print(
        "  Unsafe candidate rate: "
        f"{summary['unsafe_candidate_rate']:.1%}"
    )

    print("\nPER-CATEGORY SCREENING")

    for category in sorted(category_counts):
        category_result = category_counts[category]

        print(
            f"  {category}: "
            f"candidate={category_result['candidate']}, "
            f"blocked={category_result['blocked']}, "
            f"unresolved={category_result['unresolved']}"
        )

    print("\nUNSUPPORTED QUESTIONS MARKED AS CANDIDATES")

    unsafe = [
        record
        for record in output_records
        if (
            not record["expected_supported"]
            and record["fallback_eligibility"]
            == FallbackEligibility.CANDIDATE.value
        )
    ]

    if not unsafe:
        print("  None in this development set.")
    else:
        for record in unsafe:
            print(
                f"  {record['case_id']}: "
                f"{record['question']}"
            )

    print(f"\nSaved results: {OUTPUT_PATH}")
    print("=" * 68)


if __name__ == "__main__":
    main()
