
"""Phase 11.4G-12C: three-way hybrid router evaluation.

Replays saved Qwen proposals from Phase 11.4G-11.

This is a development-set diagnostic, not an independent
held-out evaluation or a live model inference run.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from src.recovery.hybrid_semantic_verifier import (
    HybridDecision,
    HybridSemanticVerifier,
)


INPUT_PATH = Path(
    "outputs/phase11_4g11_adversarial_v2_results.json"
)

OUTPUT_PATH = Path(
    "outputs/phase11_4g12_hybrid_router_results.json"
)


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing saved evaluation: {INPUT_PATH}"
        )

    payload = json.loads(
        INPUT_PATH.read_text(encoding="utf-8")
    )

    records = payload["records"]

    if not records:
        raise ValueError("No evaluation records found.")

    verifier = HybridSemanticVerifier()

    decisions = Counter()
    by_expected = defaultdict(Counter)
    by_category = defaultdict(Counter)
    output_records = []

    for record in records:
        question = record["question"]
        proposal = record.get("proposal")
        expected_supported = record["expected_supported"]
        category = record["category"]

        # An absent or invalid proposal cannot be accepted.
        # Preserve this distinction in the output.
        if proposal is None:
            decision = HybridDecision.REJECT
            reason = "missing_proposal"
            explanation = "Qwen did not provide a proposal."
        else:
            result = verifier.verify(question, proposal)
            decision = result.decision
            reason = result.reason.value
            explanation = result.explanation

        decision_name = decision.value

        decisions[decision_name] += 1

        group = (
            "supported"
            if expected_supported
            else "unsupported"
        )

        by_expected[group][decision_name] += 1
        by_category[category][decision_name] += 1

        output_records.append(
            {
                "case_id": record["case_id"],
                "category": category,
                "question": question,
                "expected_supported": expected_supported,
                "qwen_status": record.get("status"),
                "decision": decision_name,
                "reason": reason,
                "explanation": explanation,
            }
        )

        print(
            f"{record['case_id']}: "
            f"{group} -> {decision_name} ({reason})"
        )

    supported = sum(
        r["expected_supported"] for r in records
    )
    unsupported = len(records) - supported

    supported_accept = by_expected["supported"]["accept"]
    supported_reject = by_expected["supported"]["reject"]
    supported_abstain = by_expected["supported"]["abstain"]

    unsupported_accept = by_expected["unsupported"]["accept"]
    unsupported_reject = by_expected["unsupported"]["reject"]
    unsupported_abstain = by_expected["unsupported"]["abstain"]

    summary = {
        "total": len(records),
        "supported": supported,
        "unsupported": unsupported,
        "supported_accept": supported_accept,
        "supported_reject": supported_reject,
        "supported_abstain": supported_abstain,
        "unsupported_accept": unsupported_accept,
        "unsupported_reject": unsupported_reject,
        "unsupported_abstain": unsupported_abstain,
        "decision_counts": dict(decisions),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "experiment": "phase11_4g12c",
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
    print("PHASE 11.4G-12C — HYBRID ROUTER RESULTS")
    print(f"Total questions: {len(records)}")
    print(f"Supported: {supported}")
    print(f"Unsupported: {unsupported}")

    print("\nSUPPORTED QUESTIONS")
    print(f"  Accepted: {supported_accept}/{supported}")
    print(f"  Rejected: {supported_reject}/{supported}")
    print(f"  Abstained: {supported_abstain}/{supported}")

    print("\nUNSUPPORTED QUESTIONS")
    print(f"  Incorrectly accepted: {unsupported_accept}/{unsupported}")
    print(f"  Rejected: {unsupported_reject}/{unsupported}")
    print(f"  Abstained: {unsupported_abstain}/{unsupported}")

    print("\nPER-CATEGORY DECISIONS")

    for category in sorted(by_category):
        counts = by_category[category]
        print(
            f"  {category}: "
            f"accept={counts['accept']}, "
            f"reject={counts['reject']}, "
            f"abstain={counts['abstain']}"
        )

    print(f"\nSaved results: {OUTPUT_PATH}")
    print("=" * 68)


if __name__ == "__main__":
    main()
