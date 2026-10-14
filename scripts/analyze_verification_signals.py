
"""Phase 11.3C: analyze lexical diagnostics against offline gold outcomes."""

from __future__ import annotations

import argparse
import csv
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="docs/experiments/phase_11_3b_results.csv",
    )
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError("No evaluation rows found.")

    for row in rows:
        row["recovered"] = int(row["recovered"])
        row["lost"] = int(row["lost"])
        row["repaired"] = int(row["repaired"])
        row["harmed"] = int(row["harmed"])
        row["coverage_delta"] = (
            float(row["coverage_delta"])
            if row["coverage_delta"]
            else None
        )

    def category(row):
        delta = row["coverage_delta"]
        if delta is None:
            return "undefined"
        if delta > 1e-9:
            return "improved"
        if delta < -1e-9:
            return "decreased"
        return "unchanged"

    groups = {}
    for row in rows:
        groups.setdefault(category(row), []).append(row)

    print("\nPHASE 11.3C — VERIFICATION SIGNAL ANALYSIS")
    print(f"Total questions: {len(rows)}")

    for name in ("improved", "unchanged", "decreased", "undefined"):
        subset = groups.get(name, [])
        if not subset:
            continue

        print(f"\nCoverage: {name}")
        print(f"  Questions: {len(subset)}")
        print(
            "  Questions with gold-fact gain: "
            f"{sum(r['recovered'] > 0 for r in subset)}"
        )
        print(
            "  Fully repaired: "
            f"{sum(r['repaired'] for r in subset)}"
        )
        print(
            "  Harmed: "
            f"{sum(r['harmed'] for r in subset)}"
        )

    recovered_without_gain = [
        row for row in rows
        if row["recovered"] > 0
        and category(row) != "improved"
    ]

    improved_without_recovery = [
        row for row in rows
        if category(row) == "improved"
        and row["recovered"] == 0
    ]

    print("\nKEY DIAGNOSTICS")
    print(
        "Gold-fact gain without lexical improvement: "
        f"{len(recovered_without_gain)}"
    )
    print(
        "Lexical improvement without gold-fact gain: "
        f"{len(improved_without_recovery)}"
    )
    print(
        "Positions with gold gain but no lexical improvement: "
        f"{[r['position'] for r in recovered_without_gain]}"
    )
    print(
        "Positions with lexical improvement but no gold gain: "
        f"{[r['position'] for r in improved_without_recovery]}"
    )

    print("\nPHASE 11.3C COMPLETED")


if __name__ == "__main__":
    main()
