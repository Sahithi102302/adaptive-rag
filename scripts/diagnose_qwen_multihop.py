
"""Phase 11.4C-2: controlled multi-hop sufficiency diagnostic."""

from dataclasses import dataclass

from src.recovery.llm_semantic_assessor import LLMSemanticAssessor
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
    "Which university did the author of "
    "The Silver Atlas attend?"
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


def main():
    print("\nPHASE 11.4C-2 — MULTI-HOP DIAGNOSTIC")

    generator = QwenTextGenerator(
        QwenGeneratorConfig(max_new_tokens=256)
    )

    assessor = LLMSemanticAssessor(generator)

    results = []

    for name, evidence, expected_supported in CASES:
        print(f"\n{'=' * 60}")
        print(f"CASE: {name}")
        print(f"{'=' * 60}")

        assessment = assessor.assess(QUESTION, evidence)

        predicted_supported = (
            assessment.support == SemanticSupport.SUPPORTED
        )

        correct = predicted_supported == expected_supported

        print(f"Support: {assessment.support.value}")
        print(f"Expected SUPPORTED: {expected_supported}")
        print(f"Predicted SUPPORTED: {predicted_supported}")
        print(f"Expected classification matched: {correct}")
        print(f"Explanation: {assessment.explanation}")

        for finding in assessment.findings:
            print(
                f"Finding: {finding.finding_type.value} "
                f"| citations={finding.supporting_chunk_ids} "
                f"| {finding.explanation}"
            )

        results.append((name, correct, assessment.support.value))

    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)

    for name, correct, support in results:
        print(
            f"{name}: support={support}, "
            f"expected_classification_matched={correct}"
        )

    print(
        f"\nMatched: {sum(r[1] for r in results)}/{len(results)}"
    )
    print(
        "Diagnostic only. No recovery acceptance decisions."
    )


if __name__ == "__main__":
    main()
