from __future__ import annotations

from src.recovery.bridge_selection import (
    BridgeCandidate,
)
from src.recovery.llm_bridge_selection import (
    ClosedCandidateBridgeSelector,
    LLMBridgeSelectionConfig,
)


class FakeGenerator:
    """
    Deterministic fake generator.

    No real language model is involved in this unit test.
    """

    def __init__(
        self,
        output: str,
    ) -> None:
        self.output = output
        self.last_prompt: str | None = None

    def generate(
        self,
        prompt: str,
    ) -> str:
        self.last_prompt = prompt
        return self.output


def make_candidate(
    entity: str,
    label: str,
    source_rank: int,
    chunk_id: str,
    title: str,
    text: str,
) -> BridgeCandidate:
    return BridgeCandidate(
        entity=entity,
        entity_label=label,
        ner_score=0.99,
        source_rank=source_rank,
        source_chunk_id=chunk_id,
        source_title=title,
        source_text=text,
    )


def main() -> None:
    print("=" * 80)
    print(
        "PHASE 10.3G-D — CLOSED-CANDIDATE "
        "LLM BRIDGE-SELECTION TEST"
    )
    print("=" * 80)

    candidates = (
        make_candidate(
            entity="Example Championship",
            label="MISC",
            source_rank=1,
            chunk_id="chunk-1",
            title="Example tournament",
            text=(
                "The tournament was held in 2020. "
                "Alex Morgan was the defending champion."
            ),
        ),
        make_candidate(
            entity="Alex Morgan",
            label="PER",
            source_rank=1,
            chunk_id="chunk-1",
            title="Example tournament",
            text=(
                "The tournament was held in 2020. "
                "Alex Morgan was the defending champion."
            ),
        ),
        make_candidate(
            entity="Taylor Reed",
            label="PER",
            source_rank=2,
            chunk_id="chunk-2",
            title="Other tournament",
            text=(
                "Taylor Reed won a different competition."
            ),
        ),
    )

    # --------------------------------------------------------
    # 1. Valid closed-candidate selection
    # --------------------------------------------------------

    generator = FakeGenerator(
        '{"selected_index": 2}'
    )

    selector = ClosedCandidateBridgeSelector(
        generator=generator,
        config=LLMBridgeSelectionConfig(
            max_passage_chars=1200,
            include_ner_metadata=True,
        ),
    )

    result = selector.select(
        question=(
            "What nationality was the defending champion "
            "of the tournament?"
        ),
        candidates=candidates,
    )

    assert result.selected_index == 2

    assert (
        result.selected_candidate.entity
        == "Alex Morgan"
    )

    assert generator.last_prompt is not None

    assert (
        "Alex Morgan"
        in generator.last_prompt
    )

    assert (
        "Taylor Reed"
        in generator.last_prompt
    )

    assert (
        "Do NOT generate a retrieval query."
        in generator.last_prompt
    )

    print(
        "PASS: valid supplied candidate was selected."
    )

    # --------------------------------------------------------
    # 2. Same source passage should appear once in evidence
    # --------------------------------------------------------

    prompt = result.prompt

    assert (
        prompt.count(
            "[Retrieved passage rank 1]"
        )
        == 1
    )

    print(
        "PASS: duplicate candidate source passages were "
        "deduplicated in the evidence block."
    )

    # --------------------------------------------------------
    # 3. Optional Markdown JSON fence
    # --------------------------------------------------------

    fenced_generator = FakeGenerator(
        '```json\n'
        '{"selected_index": 1}\n'
        '```'
    )

    fenced_selector = (
        ClosedCandidateBridgeSelector(
            generator=fenced_generator
        )
    )

    fenced_result = (
        fenced_selector.select(
            question="Example question?",
            candidates=candidates,
        )
    )

    assert (
        fenced_result.selected_index
        == 1
    )

    print(
        "PASS: optional outer JSON code fence was accepted."
    )

    # --------------------------------------------------------
    # 4. Out-of-range index must fail
    # --------------------------------------------------------

    bad_index_generator = FakeGenerator(
        '{"selected_index": 99}'
    )

    bad_index_selector = (
        ClosedCandidateBridgeSelector(
            generator=bad_index_generator
        )
    )

    try:
        bad_index_selector.select(
            question="Example question?",
            candidates=candidates,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Out-of-range selected_index "
            "should have failed."
        )

    print(
        "PASS: out-of-range candidate index was rejected."
    )

    # --------------------------------------------------------
    # 5. Invented entity output must fail
    # --------------------------------------------------------

    invented_generator = FakeGenerator(
        '{"selected_entity": "Invented Person"}'
    )

    invented_selector = (
        ClosedCandidateBridgeSelector(
            generator=invented_generator
        )
    )

    try:
        invented_selector.select(
            question="Example question?",
            candidates=candidates,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Invented entity output should have failed."
        )

    print(
        "PASS: invented/free-form entity output was rejected."
    )

    # --------------------------------------------------------
    # 6. Extra JSON fields must fail
    # --------------------------------------------------------

    extra_field_generator = FakeGenerator(
        (
            '{"selected_index": 2, '
            '"reason": "best bridge"}'
        )
    )

    extra_field_selector = (
        ClosedCandidateBridgeSelector(
            generator=extra_field_generator
        )
    )

    try:
        extra_field_selector.select(
            question="Example question?",
            candidates=candidates,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Extra JSON fields should have failed."
        )

    print(
        "PASS: extra/free-form JSON fields were rejected."
    )

    # --------------------------------------------------------
    # 7. Empty candidate pool must fail
    # --------------------------------------------------------

    empty_generator = FakeGenerator(
        '{"selected_index": 1}'
    )

    empty_selector = (
        ClosedCandidateBridgeSelector(
            generator=empty_generator
        )
    )

    try:
        empty_selector.select(
            question="Example question?",
            candidates=(),
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Empty candidate pool should have failed."
        )

    print(
        "PASS: empty candidate pool was rejected."
    )

    print()
    print(
        "PHASE 10.3G-D UNIT TEST PASSED."
    )


if __name__ == "__main__":
    main()