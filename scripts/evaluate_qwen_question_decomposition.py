
"""Phase 11.4G-6: compare Qwen decomposition with the rule baseline.

Reuses the 16 hand-authored cases from Phase 11.4G-3.

This benchmark measures fidelity to the declared two-hop
author/university task scope, not unrestricted question
decomposition quality.

No retrieved evidence or gold supporting facts are provided.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scripts.evaluate_question_decomposition import CASES
from src.recovery.qwen_question_decomposer import (
    QwenQuestionDecomposer,
    QwenDecompositionStatus,
)


OUTPUT_PATH = Path(
    "outputs/phase11_4g6_qwen_decomposition_results.json"
)


def evaluate_semantics(requirements, case):
    """Compare relationship meaning, allowing variable renaming."""

    relations = requirements.relationships

    if len(relations) != 2:
        return False, "wrong_relationship_count"

    author_links = [
        relation
        for relation in relations
        if relation.predicate == "authored_by"
    ]

    education_links = [
        relation
        for relation in relations
        if relation.predicate == "attended"
    ]

    if len(author_links) != 1 or len(education_links) != 1:
        return False, "wrong_relationship"

    author_link = author_links[0]
    education_link = education_links[0]

    if author_link.subject != case.expected_title:
        return False, "wrong_entity"

    if not author_link.object.startswith("?"):
        return False, "missing_author_variable"

    if education_link.subject != author_link.object:
        return False, "broken_variable_join"

    if not education_link.object.startswith("?"):
        return False, "missing_answer_variable"

    if requirements.answer_variable != education_link.object:
        return False, "wrong_answer_target"

    if requirements.answer_variable == author_link.object:
        return False, "answer_targets_author"

    return True, "correct"


def classify(case, result):
    valid = result.status == QwenDecompositionStatus.VALID_PROPOSAL

    if not valid:
        if case.expected_supported:
            return "missed_supported"

        return "correct_rejection"

    requirements = result.validation.requirements

    if not case.expected_supported:
        return "false_acceptance"

    correct, reason = evaluate_semantics(requirements, case)

    if correct:
        return "correct_decomposition"

    return f"incorrect_decomposition:{reason}"


def main():
    print("Loading Qwen once for the 16-question evaluation...")
    decomposer = QwenQuestionDecomposer()

    counts = Counter()
    records = []

    for index, case in enumerate(CASES, start=1):
        print("\n" + "=" * 70, flush=True)
        print(
            f"[{index}/{len(CASES)}] {case.case_id}: "
            f"{case.question}",
            flush=True,
        )

        result = decomposer.decompose(case.question)
        outcome = classify(case, result)

        counts[outcome] += 1
        counts[result.status.value] += 1

        print(f"Expected supported: {case.expected_supported}")
        print(f"Qwen status: {result.status.value}")
        print(f"Outcome: {outcome}")
        print(f"Raw response:\n{result.raw_response}")

        record = {
            "case_id": case.case_id,
            "question": case.question,
            "expected_supported": case.expected_supported,
            "expected_title": case.expected_title,
            "qwen_status": result.status.value,
            "outcome": outcome,
            "raw_response": result.raw_response,
            "proposal": result.proposal,
            "explanation": result.explanation,
        }

        records.append(record)

        # Save after every question, so a later interruption
        # does not erase completed diagnostic results.
        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        OUTPUT_PATH.write_text(
            json.dumps(
                {
                    "experiment": "phase11_4g6",
                    "completed_cases": len(records),
                    "total_cases": len(CASES),
                    "records": records,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    supported = sum(
        case.expected_supported for case in CASES
    )
    unsupported = len(CASES) - supported

    correct = counts["correct_decomposition"]
    rejected = counts["correct_rejection"]
    missed = counts["missed_supported"]
    false_acceptances = counts["false_acceptance"]

    incorrect = sum(
        count
        for key, count in counts.items()
        if key.startswith("incorrect_decomposition:")
    )

    coverage = correct / supported if supported else 0.0
    specificity = (
        rejected / unsupported if unsupported else 0.0
    )

    print("\n" + "=" * 70)
    print("PHASE 11.4G-6 — QWEN DECOMPOSITION EVALUATION")
    print(f"Cases: {len(CASES)}")
    print(f"Expected supported: {supported}")
    print(f"Expected unsupported: {unsupported}")
    print(f"Correct decompositions: {correct}")
    print(f"Correct rejections: {rejected}")
    print(f"Missed supported: {missed}")
    print(f"Incorrect decompositions: {incorrect}")
    print(f"False acceptances: {false_acceptances}")
    print(
        "Correct decisions: "
        f"{correct + rejected}/{len(CASES)}"
    )
    print(f"Supported-question coverage: {coverage:.1%}")
    print(
        "Unsupported-question rejection specificity: "
        f"{specificity:.1%}"
    )

    print("\nQwen response statuses:")
    for status in QwenDecompositionStatus:
        print(
            f"  {status.value}: "
            f"{counts[status.value]}"
        )

    print("\nDeterministic baseline:")
    print("  Correct decompositions: 4/9")
    print("  Correct rejections: 7/7")
    print("  Overall correct decisions: 11/16")

    print(f"\nSaved results: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
