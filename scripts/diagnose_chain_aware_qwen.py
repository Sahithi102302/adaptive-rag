
"""Phase 11.4D-3: real-Qwen chain-aware diagnostic.

Compare chain-aware verification against the original
five-case synthetic multi-hop evaluation.

Diagnostic only. No gold annotations at runtime,
recovery decisions, or automatic answer acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.recovery.chain_aware_assessor import (
    ChainAwareSemanticAssessor,
)
from src.recovery.qwen_generator import (
    QwenGeneratorConfig,
    QwenTextGenerator,
)
from src.recovery.semantic_verification import SemanticSupport


@dataclass(frozen=True)
class FakeChunk:
    id: str
    text: str


@dataclass(frozen=True)
class FakeResult:
    chunk: FakeChunk


def make_evidence(*pairs):
    return tuple(
        FakeResult(FakeChunk(chunk_id, text))
        for chunk_id, text in pairs
    )


QUESTION = (
    "Which university did the author "
    "of The Silver Atlas attend?"
)


CASES = [
    (
        "A_COMPLETE",
        make_evidence(
            ("book", "The Silver Atlas was written by Elena Marsh."),
            ("education", "Elena Marsh attended Northbridge University."),
        ),
        True,
    ),
    (
        "B_MISSING_SECOND_HOP",
        make_evidence(
            ("book", "The Silver Atlas was written by Elena Marsh."),
        ),
        False,
    ),
    (
        "C_MISSING_FIRST_HOP",
        make_evidence(
            ("education", "Elena Marsh attended Northbridge University."),
        ),
        False,
    ),
    (
        "D_DISCONNECTED",
        make_evidence(
            ("book", "The Silver Atlas was written by Elena Marsh."),
            ("education", "Daniel Reed attended Northbridge University."),
        ),
        False,
    ),
    (
        "E_CONTRADICTION",
        make_evidence(
            ("book", "The Silver Atlas was written by Elena Marsh."),
            ("education_a", "Elena Marsh attended Northbridge University."),
            ("education_b", "Elena Marsh never attended Northbridge University."),
        ),
        False,
    ),
]


class RecordingGenerator:
    """Records model output without changing generation behavior."""

    def __init__(self, generator):
        self.generator = generator
        self.last_response = None

    def generate(self, prompt):
        self.last_response = None
        response = self.generator.generate(prompt)
        self.last_response = response
        return response


def main():
    print("\nPHASE 11.4D-3 — CHAIN-AWARE QWEN DIAGNOSTIC")

    # Longer output allowance for structured reasoning chains.
    generator = RecordingGenerator(
        QwenTextGenerator(
            QwenGeneratorConfig(max_new_tokens=512)
        )
    )

    assessor = ChainAwareSemanticAssessor(generator)

    results = []

    for name, evidence, expected_supported in CASES:
        print("\n" + "=" * 65)
        print(f"CASE: {name}")
        print("=" * 65)

        assessment = assessor.assess(QUESTION, evidence)

        predicted_supported = (
            assessment.support == SemanticSupport.SUPPORTED
        )

        matched = predicted_supported == expected_supported

        unavailable = assessment.explanation.startswith(
            "Chain assessment unavailable:"
        )

        print(f"\nSupport: {assessment.support.value}")
        print(f"Candidate answer: {assessment.candidate_answer}")
        print(f"Expected SUPPORTED: {expected_supported}")
        print(f"Predicted SUPPORTED: {predicted_supported}")
        print(f"Classification matched: {matched}")
        print(f"Assessment unavailable: {unavailable}")
        print(f"Explanation: {assessment.explanation}")

        print("\nREASONING STEPS")

        if not assessment.steps:
            print("No validated reasoning steps.")

        for step in assessment.steps:
            print(f"Step {step.step_id}: {step.relationship}")
            print(f"  Status: {step.status.value}")
            print(f"  Citations: {step.supporting_chunk_ids}")

        print("\nCONTRADICTIONS")

        if not assessment.contradictions:
            print("None reported.")

        for contradiction in assessment.contradictions:
            print(f"  {contradiction.explanation}")
            print(f"  Citations: {contradiction.supporting_chunk_ids}")

        # Synthetic evidence only: print raw model output
        # when validation fails, to diagnose formatting.
        if unavailable:
            print("\nRAW MODEL RESPONSE")
            print(repr(generator.last_response))

        results.append(
            (name, assessment.support.value, matched, unavailable)
        )

    print("\n" + "=" * 65)
    print("FINAL DIAGNOSTIC SUMMARY")
    print("=" * 65)

    for name, support, matched, unavailable in results:
        print(
            f"{name}: support={support}, "
            f"matched={matched}, unavailable={unavailable}"
        )

    matched_count = sum(item[2] for item in results)
    unavailable_count = sum(item[3] for item in results)

    print(f"\nExpected classifications matched: {matched_count}/5")
    print(f"Unavailable assessments: {unavailable_count}/5")
    print("Original assessor matched: 1/5")

    print("\nDiagnostic only. No recovery acceptance.")


if __name__ == "__main__":
    main()
