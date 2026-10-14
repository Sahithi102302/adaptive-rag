
"""Phase 11.4E-4: real-Qwen independent chain verification.

Diagnostic only. Does not authorize recovery acceptance.
"""

from __future__ import annotations

from src.recovery.chain_aware_assessor import (
    ChainAwareSemanticAssessor,
)
from src.recovery.chain_entailment_verifier import (
    ChainEntailmentVerifier,
)
from src.recovery.qwen_generator import (
    QwenGeneratorConfig,
    QwenTextGenerator,
)
from src.recovery.semantic_verification import SemanticSupport

from scripts.diagnose_chain_aware_qwen import (
    CASES,
    QUESTION,
)


def main():
    print("\nPHASE 11.4E-4 — REAL-QWEN CHAIN VERIFICATION")

    generator = QwenTextGenerator(
        QwenGeneratorConfig(max_new_tokens=512)
    )

    proposer = ChainAwareSemanticAssessor(generator)
    verifier = ChainEntailmentVerifier()

    results = []

    for name, evidence, expected_supported in CASES:
        print("\n" + "=" * 65)
        print(f"CASE: {name}")
        print("=" * 65)

        evidence_by_id = {
            item.chunk.id: item.chunk.text
            for item in evidence
        }

        proposed = proposer.assess(
            question=QUESTION,
            evidence=evidence,
        )

        verified = verifier.verify(
            chain=proposed,
            evidence_by_id=evidence_by_id,
        )

        proposed_supported = (
            proposed.support == SemanticSupport.SUPPORTED
        )

        literal_gate_passed = verified.passes_literal_gate

        matched = (
            literal_gate_passed == expected_supported
        )

        unavailable = proposed.explanation.startswith(
            "Chain assessment unavailable:"
        )

        print(f"\nQwen support: {proposed.support.value}")
        print(f"Qwen candidate answer: {proposed.candidate_answer}")
        print(f"Assessment unavailable: {unavailable}")
        print(f"Original proposed SUPPORTED: {proposed_supported}")
        print(f"Literal gate passed: {literal_gate_passed}")
        print(f"Expected SUPPORTED: {expected_supported}")
        print(f"Gate classification matched: {matched}")

        print("\nINDEPENDENT CLAIM VERIFICATION")

        if not verified.assessment.claim_verifications:
            print("No proposed claims available.")

        for index, claim_result in enumerate(
            verified.assessment.claim_verifications,
            start=1,
        ):
            print(f"\nClaim {index}: {claim_result.claim}")
            print(f"  Label: {claim_result.label.value}")
            print(
                f"  Citations: {claim_result.cited_chunk_ids}"
            )
            print(
                f"  Explanation: {claim_result.explanation}"
            )

        print("\nCHAIN-LEVEL RESULT")
        print(verified.explanation)

        results.append(
            {
                "name": name,
                "proposed_supported": proposed_supported,
                "gate_passed": literal_gate_passed,
                "expected_supported": expected_supported,
                "matched": matched,
                "unavailable": unavailable,
            }
        )

    print("\n" + "=" * 65)
    print("FINAL DIAGNOSTIC SUMMARY")
    print("=" * 65)

    for result in results:
        print(
            f"{result['name']}: "
            f"qwen_supported={result['proposed_supported']}, "
            f"literal_gate={result['gate_passed']}, "
            f"expected={result['expected_supported']}, "
            f"matched={result['matched']}, "
            f"unavailable={result['unavailable']}"
        )

    matched_count = sum(
        result["matched"] for result in results
    )

    false_acceptances = sum(
        result["gate_passed"]
        and not result["expected_supported"]
        for result in results
    )

    false_rejections = sum(
        not result["gate_passed"]
        and result["expected_supported"]
        for result in results
    )

    unavailable_count = sum(
        result["unavailable"] for result in results
    )

    print(f"\nGate classifications matched: {matched_count}/5")
    print(f"False acceptances: {false_acceptances}/4 negatives")
    print(f"False rejections: {false_rejections}/1 positive")
    print(f"Unavailable Qwen assessments: {unavailable_count}/5")
    print("Previous Qwen-only diagnostic: 1/5")

    print(
        "\nIMPORTANT: Literal gate results do not establish "
        "chain completeness or contradiction coverage."
    )
    print("Diagnostic only. No automatic recovery acceptance.")


if __name__ == "__main__":
    main()
