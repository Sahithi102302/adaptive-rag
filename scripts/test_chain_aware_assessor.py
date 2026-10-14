
"""Phase 11.4D-2 tests without loading Qwen."""

import json
from dataclasses import dataclass

from src.recovery.chain_aware_assessor import (
    ChainAwareSemanticAssessor,
)
from src.recovery.semantic_verification import SemanticSupport


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
                "book",
                "The Silver Atlas was written by Elena Marsh.",
            )
        ),
        FakeResult(
            FakeChunk(
                "education",
                "Elena Marsh attended Northbridge University.",
            )
        ),
    )


QUESTION = (
    "Which university did the author "
    "of The Silver Atlas attend?"
)


def valid_payload():
    return {
        "candidate_answer": "Northbridge University",
        "steps": [
            {
                "step_id": 1,
                "relationship": (
                    "The Silver Atlas was written by Elena Marsh."
                ),
                "status": "supported",
                "supporting_chunk_ids": ["book"],
            },
            {
                "step_id": 2,
                "relationship": (
                    "Elena Marsh attended Northbridge University."
                ),
                "status": "supported",
                "supporting_chunk_ids": ["education"],
            },
        ],
        "contradictions": [],
        "support": "supported",
        "explanation": "Both required links are supported.",
    }


def assess(payload, supplied_evidence=None):
    return ChainAwareSemanticAssessor(
        FakeGenerator(json.dumps(payload))
    ).assess(
        QUESTION,
        evidence() if supplied_evidence is None else supplied_evidence,
    )


def test_valid_complete_chain():
    result = assess(valid_payload())

    assert result.support == SemanticSupport.SUPPORTED
    assert result.candidate_answer == "Northbridge University"
    assert len(result.steps) == 2


def test_fenced_json():
    raw = "```json\n" + json.dumps(valid_payload()) + "\n```"

    result = ChainAwareSemanticAssessor(
        FakeGenerator(raw)
    ).assess(QUESTION, evidence())

    assert result.support == SemanticSupport.SUPPORTED


def test_missing_second_hop():
    payload = valid_payload()
    payload["candidate_answer"] = None
    payload["steps"][1]["status"] = "missing"
    payload["steps"][1]["supporting_chunk_ids"] = []
    payload["support"] = "partial"

    result = assess(payload)

    assert result.support == SemanticSupport.PARTIAL
    assert result.has_missing_steps


def test_supported_with_missing_step_rejected():
    payload = valid_payload()
    payload["steps"][1]["status"] = "missing"
    payload["steps"][1]["supporting_chunk_ids"] = []

    result = assess(payload)

    assert result.support == SemanticSupport.UNKNOWN
    assert result.steps == ()


def test_unknown_citation_rejected():
    payload = valid_payload()
    payload["steps"][0]["supporting_chunk_ids"] = ["invented"]

    result = assess(payload)

    assert result.support == SemanticSupport.UNKNOWN


def test_supported_without_answer_rejected():
    payload = valid_payload()
    payload["candidate_answer"] = None

    result = assess(payload)

    assert result.support == SemanticSupport.UNKNOWN


def test_invalid_json_rejected():
    result = ChainAwareSemanticAssessor(
        FakeGenerator("not JSON")
    ).assess(QUESTION, evidence())

    assert result.support == SemanticSupport.UNKNOWN


def test_extra_prose_rejected():
    raw = "Here is the result:\n" + json.dumps(valid_payload())

    result = ChainAwareSemanticAssessor(
        FakeGenerator(raw)
    ).assess(QUESTION, evidence())

    assert result.support == SemanticSupport.UNKNOWN


def test_empty_evidence_skips_generation():
    generator = FakeGenerator(json.dumps(valid_payload()))

    result = ChainAwareSemanticAssessor(generator).assess(
        QUESTION, ()
    )

    assert result.support == SemanticSupport.UNKNOWN
    assert not generator.prompts


def test_duplicate_evidence_rejected():
    repeated = (evidence()[0], evidence()[0])

    try:
        ChainAwareSemanticAssessor(
            FakeGenerator(json.dumps(valid_payload()))
        ).assess(QUESTION, repeated)
    except ValueError:
        return

    raise AssertionError("Expected duplicate evidence rejection.")


def test_conflict_blocks_supported():
    payload = valid_payload()
    payload["steps"][1]["status"] = "conflicting"
    payload["steps"][1]["supporting_chunk_ids"] = [
        "education", "negative"
    ]
    payload["contradictions"] = [
        {
            "explanation": "The passages disagree.",
            "supporting_chunk_ids": ["education", "negative"],
        }
    ]

    extended = evidence() + (
        FakeResult(
            FakeChunk(
                "negative",
                "Elena Marsh never attended Northbridge University.",
            )
        ),
    )

    result = assess(payload, extended)

    assert result.support == SemanticSupport.UNKNOWN


def main():
    tests = [
        test_valid_complete_chain,
        test_fenced_json,
        test_missing_second_hop,
        test_supported_with_missing_step_rejected,
        test_unknown_citation_rejected,
        test_supported_without_answer_rejected,
        test_invalid_json_rejected,
        test_extra_prose_rejected,
        test_empty_evidence_skips_generation,
        test_duplicate_evidence_rejected,
        test_conflict_blocks_supported,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(
        "\nPHASE 11.4D-2 CHAIN-AWARE ASSESSOR "
        "TESTS PASSED"
    )
    print(f"Tests passed: {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
