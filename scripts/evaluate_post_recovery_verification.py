
"""Phase 11.3B: paired post-recovery verification diagnostics.

Uses the Phase 10.4E frozen validation retrieval setup.
Gold annotations are accessed only after candidate selection.
This is NOT a production verification or acceptance policy.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scripts.build_scaled_diagnostic_dataset import (
    get_gold_evidence,
    load_query_examples,
)
from scripts.evaluate_broad_evidence_selection import create_stack
from scripts.evaluate_expanded_candidate_recovery import (
    get_retrieved_evidence,
)
from scripts.evaluate_real_evidence_selection import (
    FINAL_K,
    build_bridge_candidates,
)

from src.recovery.evidence_selection import EvidencePreservingSelector
from src.recovery.offline_verification import (
    evaluate_offline_verification,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-questions", type=int, default=50)
    parser.add_argument(
        "--output",
        default="docs/experiments/phase_11_3b_results.csv",
    )
    args = parser.parse_args()

    if not 1 <= args.max_questions <= 200:
        parser.error("--max-questions must be between 1 and 200")

    examples = load_query_examples("validation")

    (
        embedder,
        reranker,
        second_hop,
        extractor,
        bridge_selector,
        query_generator,
    ) = create_stack()

    evidence_selector = EvidencePreservingSelector()
    rows = []

    print("\nPHASE 11.3B — POST-RECOVERY VERIFICATION DIAGNOSTICS")
    print("Offline forced recovery; no production acceptance decisions.")

    for position, example in enumerate(
        examples[:args.max_questions], start=1
    ):
        first = tuple(
            reranker.search(
                query=example.question,
                embedder=embedder,
                top_k=FINAL_K,
            )
        )

        second = ()
        status = "no_candidates"

        try:
            entities = extractor.extract_from_results(first)
            candidates = build_bridge_candidates(
                entities=entities,
                results=first,
            )

            if candidates:
                selected_bridge = bridge_selector.select(
                    question=example.question,
                    candidates=candidates,
                ).selected_candidate

                generated = query_generator.generate(
                    question=example.question,
                    selected_candidate=selected_bridge,
                    candidates=candidates,
                )

                retrieval = second_hop.retrieve(
                    query=generated.query,
                    first_hop_results=first,
                )

                if retrieval.first_hop_results != first:
                    raise ValueError("First-hop provenance mismatch.")

                second = tuple(retrieval.second_hop_results)
                status = "retrieved"

        except ValueError as exc:
            status = "invalid_query_or_bridge"
            print(f"Position {position}: {exc}", flush=True)

        selection = evidence_selector.select(first, second)
        candidate = selection.selected_results

        # Candidate selection and lexical assessment do not use gold.
        # The offline evaluator receives gold only for scoring.
        gold = get_gold_evidence(example)

        if not gold:
            continue

        outcome = evaluate_offline_verification(
            query=example.question,
            original_results=first,
            candidate_results=candidate,
            gold_evidence=gold,
            evidence_extractor=get_retrieved_evidence,
        )

        comparison = outcome.comparison

        rows.append(
            {
                "position": position,
                "status": status,
                "original_gold": outcome.original_gold_count,
                "candidate_gold": outcome.candidate_gold_count,
                "gold_total": outcome.gold_total,
                "recovered": outcome.recovered_count,
                "lost": outcome.lost_count,
                "originally_complete": int(outcome.originally_complete),
                "candidate_complete": int(outcome.candidate_complete),
                "repaired": int(outcome.repaired),
                "harmed": int(outcome.harmed),
                "complete_to_incomplete": int(
                    outcome.complete_to_incomplete
                ),
                "original_coverage": (
                    comparison.original.query_token_coverage
                ),
                "candidate_coverage": (
                    comparison.candidate.query_token_coverage
                ),
                "coverage_delta": comparison.coverage_delta,
                "gained_token_count": len(
                    comparison.gained_query_tokens
                ),
                "lost_token_count": len(
                    comparison.lost_query_tokens
                ),
                "new_document_count": comparison.new_document_count,
                "retained_chunk_count": comparison.retained_chunk_count,
                "second_hop_count": len(second),
            }
        )

        print(
            f"[{position}/{args.max_questions}] "
            f"status={status} "
            f"recovered={outcome.recovered_count} "
            f"lost={outcome.lost_count} "
            f"coverage_delta={comparison.coverage_delta}",
            flush=True,
        )

    if not rows:
        raise RuntimeError("No evaluable validation examples.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nEvaluated {len(rows)} questions.")
    print(f"Repaired cases: {sum(row['repaired'] for row in rows)}")
    print(f"Harmed cases: {sum(row['harmed'] for row in rows)}")
    print(
        "Lexical coverage improved: "
        f"{sum((row['coverage_delta'] or 0) > 0 for row in rows)}"
    )
    print(f"Results saved: {output}")
    print("\nPHASE 11.3B COMPLETED")


if __name__ == "__main__":
    main()
