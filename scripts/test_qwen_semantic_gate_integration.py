
"""Phase 11.4G-8: test semantic gating of saved Qwen proposals.

No model loading or new inference.
Replays proposals saved during Phase 11.4G-6.
"""

import json
from pathlib import Path

from src.recovery.decomposition_proposal_validator import (
    DecompositionProposalValidator,
)
from src.recovery.decomposition_semantic_gate import (
    DecompositionSemanticGate,
)


INPUT_PATH = Path(
    "outputs/phase11_4g6_qwen_decomposition_results.json"
)


def main():
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Previous evaluation results not found: {INPUT_PATH}"
        )

    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    records = data["records"]

    assert len(records) == 16, (
        f"Expected 16 records, received {len(records)}."
    )

    validator = DecompositionProposalValidator()
    gate = DecompositionSemanticGate()

    correct_supported = 0
    correct_rejections = 0
    false_acceptances = 0
    missed_supported = 0

    for record in records:
        question = record["question"]
        expected_supported = record["expected_supported"]
        proposal = record["proposal"]

        validation = validator.validate(question, proposal)

        if validation.is_valid:
            semantic_result = gate.check(
                question,
                validation.requirements,
            )
            accepted = semantic_result.accepted
            reason = semantic_result.reason
        else:
            accepted = False
            reason = validation.explanation

        if expected_supported and accepted:
            correct_supported += 1
            outcome = "ACCEPTED_SUPPORTED"

        elif expected_supported and not accepted:
            missed_supported += 1
            outcome = "MISSED_SUPPORTED"

        elif not expected_supported and accepted:
            false_acceptances += 1
            outcome = "FALSE_ACCEPTANCE"

        else:
            correct_rejections += 1
            outcome = "CORRECT_REJECTION"

        print("-" * 65)
        print(f"Case: {record['case_id']}")
        print(f"Outcome: {outcome}")
        print(f"Reason: {reason}")

    print("\n" + "=" * 65)
    print("PHASE 11.4G-8 — SAVED QWEN PROPOSAL GATE REGRESSION")
    print(f"Correct supported: {correct_supported}/9")
    print(f"Correct rejections: {correct_rejections}/7")
    print(f"False acceptances: {false_acceptances}")
    print(f"Missed supported: {missed_supported}")
    print(
        "Overall correct: "
        f"{correct_supported + correct_rejections}/16"
    )
    print("=" * 65)

    assert correct_supported == 9
    assert correct_rejections == 7
    assert false_acceptances == 0
    assert missed_supported == 0

    print("RESULT: PASS")


if __name__ == "__main__":
    main()
