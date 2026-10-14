
"""Phase 11.4C unit tests without loading Qwen."""

import json
from dataclasses import dataclass

from src.recovery.llm_semantic_assessor import (
    LLMSemanticAssessor,
)
from src.recovery.semantic_verification import (
    SemanticSupport,
)


@dataclass(frozen=True)
class FakeChunk:
    id: str
    text: str


@dataclass(frozen=True)
class FakeResult:
    chunk: FakeChunk


class FakeGenerator:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.response


def evidence():
    return (
        FakeResult(
            FakeChunk(
                "a",
                "Book X was written by Person Y.",
            )
        ),
        FakeResult(
            FakeChunk(
                "b",
                "Person Y attended University Z.",
            )
        ),
    )


def valid_response():
    return json.dumps(
        {
            "support": "supported",
            "explanation": "Both relationships are represented.",
            "findings": [
                {
                    "finding_type": "support",
                    "explanation": (
                        "The book's author is identified."
                    ),
                    "supporting_chunk_ids": ["a"],
                },
                {
                    "finding_type": "support",
                    "explanation": (
                        "The author's university is identified."
                    ),
                    "supporting_chunk_ids": ["b"],
                },
            ],
        }
    )


def test_valid_structured_assessment():
    generator = FakeGenerator(valid_response())

    result = LLMSemanticAssessor(generator).assess(
        "Which university did the author of Book X attend?",
        evidence(),
    )

    assert result.support == SemanticSupport.SUPPORTED
    assert result.assessed_chunk_ids == ("a", "b")
    assert len(generator.prompts) == 1


def test_invalid_json_is_unknown():
    result = LLMSemanticAssessor(
        FakeGenerator("not valid JSON")
    ).assess(
        "Question?",
        evidence(),
    )

    assert result.support == SemanticSupport.UNKNOWN


def test_unknown_citation_is_unknown():
    response = json.loads(valid_response())

    response["findings"][0][
        "supporting_chunk_ids"
    ] = ["invented"]

    result = LLMSemanticAssessor(
        FakeGenerator(json.dumps(response))
    ).assess(
        "Question?",
        evidence(),
    )

    assert result.support == SemanticSupport.UNKNOWN


def test_supported_without_finding_is_unknown():
    response = json.dumps(
        {
            "support": "supported",
            "explanation": "Claimed support.",
            "findings": [],
        }
    )

    result = LLMSemanticAssessor(
        FakeGenerator(response)
    ).assess(
        "Question?",
        evidence(),
    )

    assert result.support == SemanticSupport.UNKNOWN


def test_empty_evidence_skips_generation():
    generator = FakeGenerator(valid_response())

    result = LLMSemanticAssessor(generator).assess(
        "Question?",
        (),
    )

    assert result.support == SemanticSupport.UNKNOWN
    assert not generator.prompts


def test_duplicate_evidence_rejected():
    repeated = (
        evidence()[0],
        evidence()[0],
    )

    try:
        LLMSemanticAssessor(
            FakeGenerator(valid_response())
        ).assess(
            "Question?",
            repeated,
        )
    except ValueError:
        return

    raise AssertionError(
        "Expected duplicate evidence rejection."
    )


def test_excess_evidence_rejected():
    try:
        LLMSemanticAssessor(
            FakeGenerator(valid_response()),
            max_chunks=1,
        ).assess(
            "Question?",
            evidence(),
        )
    except ValueError:
        return

    raise AssertionError(
        "Expected evidence limit rejection."
    )


def test_fenced_json_is_accepted():
    response = (
        "```json\n"
        + valid_response()
        + "\n```"
    )

    result = LLMSemanticAssessor(
        FakeGenerator(response)
    ).assess(
        "Which university did the author of Book X attend?",
        evidence(),
    )

    assert result.support == SemanticSupport.SUPPORTED
    assert result.assessed_chunk_ids == ("a", "b")


def test_extra_prose_is_rejected():
    response = (
        "Here is my assessment:\n"
        + valid_response()
    )

    result = LLMSemanticAssessor(
        FakeGenerator(response)
    ).assess(
        "Which university did the author of Book X attend?",
        evidence(),
    )

    assert result.support == SemanticSupport.UNKNOWN


def main():
    tests = [
        test_valid_structured_assessment,
        test_invalid_json_is_unknown,
        test_unknown_citation_is_unknown,
        test_supported_without_finding_is_unknown,
        test_empty_evidence_skips_generation,
        test_duplicate_evidence_rejected,
        test_excess_evidence_rejected,
        test_fenced_json_is_accepted,
        test_extra_prose_is_rejected,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(
        "\nPHASE 11.4C LLM SEMANTIC ASSESSOR "
        "TESTS PASSED"
    )


if __name__ == "__main__":
    main()
