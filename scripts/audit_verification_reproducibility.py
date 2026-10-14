
"""Phase 11.3D: audit Phase 11 outcomes against Phase 10.4E.

Compares the same validation positions and hop-balanced 5/5 policy.
Does not rerun retrieval, generation, or use gold at runtime.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = (
    ("status", "status"),
    ("gold_total", "gold_total"),
    ("original_gold", "original_gold"),
    ("final_gold", "candidate_gold"),
    ("recovered", "recovered"),
    ("lost", "lost"),
    ("originally_complete", "originally_complete"),
    ("final_complete", "candidate_complete"),
    ("repaired", "repaired"),
    ("harmed", "harmed"),
    ("complete_to_incomplete", "complete_to_incomplete"),
    ("second_hop_count", "second_hop_count"),
)


def load_rows(path: str):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit(phase10_rows, phase11_rows):
    old = [
        row for row in phase10_rows
        if row["policy"] == "Hop-balanced 5/5"
    ]

    old_by_position = {int(row["position"]): row for row in old}
    new_by_position = {
        int(row["position"]): row for row in phase11_rows
    }

    if len(old_by_position) != len(old):
        raise ValueError("Duplicate Phase 10 positions.")

    if len(new_by_position) != len(phase11_rows):
        raise ValueError("Duplicate Phase 11 positions.")

    if set(old_by_position) != set(new_by_position):
        raise ValueError(
            "Phase 10 and Phase 11 position sets differ."
        )

    mismatches = []

    for position in sorted(old_by_position):
        old_row = old_by_position[position]
        new_row = new_by_position[position]

        for old_field, new_field in FIELDS:
            if old_row[old_field] != new_row[new_field]:
                mismatches.append(
                    (
                        position,
                        old_field,
                        old_row[old_field],
                        new_row[new_field],
                    )
                )

    return mismatches, len(old_by_position)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase10",
        default="docs/experiments/phase_10_4e_results.csv",
    )
    parser.add_argument(
        "--phase11",
        default="docs/experiments/phase_11_3b_results.csv",
    )
    args = parser.parse_args()

    mismatches, count = audit(
        load_rows(args.phase10),
        load_rows(args.phase11),
    )

    print("\nPHASE 11.3D — REPRODUCIBILITY AUDIT")
    print(f"Compared questions: {count}")
    print(f"Compared fields per question: {len(FIELDS)}")
    print(f"Total field comparisons: {count * len(FIELDS)}")
    print(f"Mismatches: {len(mismatches)}")

    for position, field, old, new in mismatches[:30]:
        print(
            f"Position {position}: {field} "
            f"Phase10={old!r} Phase11={new!r}"
        )

    if mismatches:
        raise AssertionError(
            "Cross-experiment reproducibility audit failed."
        )

    print("\nPHASE 11.3D REPRODUCIBILITY AUDIT PASSED")


if __name__ == "__main__":
    main()
