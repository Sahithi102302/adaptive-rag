
"""Phase 11.4C-Real: controlled Qwen semantic assessment smoke test.

Synthetic evidence only. No gold annotations or recovery decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.recovery.llm_semantic_assessor import LLMSemanticAssessor
from src.recovery.qwen_generator import (
    QwenGeneratorConfig,
    QwenTextGenerator,
)


@dataclass(frozen=True)
class FakeChunk:
    id: str
    text: str


@dataclass(frozen=True)
class FakeResult:
    chunk: FakeChunk


def evidence(chunk_id: str, text: str) -> FakeResult:
    return FakeResult(FakeChunk(chunk_id, text))


def show_assessment(label, assessment):
    print(f"\n{label}")
    print(f"Support: {assessment.support.value}")
    print(f"Explanation: {assessment.explanation}")
    print(f"Assessed chunks: {assessment.assessed_chunk_ids}")

    for finding in assessment.findings:
        print(
            f"  {finding.finding_type.value}: "
            f"{finding.explanation} "
            f"| citations={finding.supporting_chunk_ids}"
        )


def main():
    print("\nPHASE 11.4C-REAL — QWEN SEMANTIC SMOKE TEST")

    generator = QwenTextGenerator(
        config=QwenGeneratorConfig(max_new_tokens=256)
    )
    assessor = LLMSemanticAssessor(generator=generator)

    question = "Which university did the author of The Silver Atlas attend?"

    complete = (
        evidence(
            "book_fact",
            "The Silver Atlas was written by Elena Marsh.",
        ),
        evidence(
            "education_fact",
            "Elena Marsh attended Northbridge University.",
        ),
    )

    incomplete = (
        evidence(
            "book_fact",
            "The Silver Atlas was written by Elena Marsh.",
        ),
    )

    complete_result = assessor.assess(question, complete)
    incomplete_result = assessor.assess(question, incomplete)

    show_assessment("COMPLETE EVIDENCE", complete_result)
    show_assessment("INCOMPLETE EVIDENCE", incomplete_result)

    print("\nPHASE 11.4C-REAL SMOKE TEST FINISHED")
    print("Diagnostic only; no acceptance decisions.")


if __name__ == "__main__":
    main()
