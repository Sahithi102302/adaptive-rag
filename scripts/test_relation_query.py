from __future__ import annotations

from src.recovery.bridge_selection import (
    BridgeCandidate,
)
from src.recovery.relation_query import (
    ConstrainedRelationQueryGenerator,
    RelationQueryConfig,
)


class FakeGenerator:
    """
    Deterministic fake text generator for unit tests.
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
    source_rank: int,
    source_chunk_id: str,
    source_title: str,
    source_text: str,
    entity_label: str = "PER",
    ner_score: float = 0.99,
) -> BridgeCandidate:
    return BridgeCandidate(
        entity=entity,
        entity_label=entity_label,
        ner_score=ner_score,
        source_rank=source_rank,
        source_chunk_id=source_chunk_id,
        source_title=source_title,
        source_text=source_text,
    )


def main() -> None:
    print("=" * 80)
    print(
        "PHASE 10.3H-A — CONSTRAINED "
        "RELATION/QUERY FORMULATION TEST"
    )
    print("=" * 80)

    question = (
        "What nationality was the defending "
        "champion of the tournament?"
    )

    source_text = (
        "The tournament was held in 2013. "
        "Alex Morgan was the defending champion."
    )

    tournament = make_candidate(
        entity="Example Championship",
        source_rank=1,
        source_chunk_id="chunk_001",
        source_title="Example Championship",
        source_text=source_text,
        entity_label="MISC",
    )

    alex = make_candidate(
        entity="Alex Morgan",
        source_rank=1,
        source_chunk_id="chunk_001",
        source_title="Example Championship",
        source_text=source_text,
    )

    unrelated = make_candidate(
        entity="Taylor Reed",
        source_rank=2,
        source_chunk_id="chunk_002",
        source_title="Other Event",
        source_text=(
            "Taylor Reed competed in another event."
        ),
    )

    candidates = (
        tournament,
        alex,
        unrelated,
    )

    # --------------------------------------------------------
    # 1. Valid constrained query
    # --------------------------------------------------------

    fake = FakeGenerator(
        '{"query": "What nationality is Alex Morgan?"}'
    )

    formulator = (
        ConstrainedRelationQueryGenerator(
            generator=fake,
            config=RelationQueryConfig(
                max_passage_chars=1200,
            ),
        )
    )

    result = formulator.generate(
        question=question,
        selected_candidate=alex,
        candidates=candidates,
    )

    assert (
        result.query
        == "What nationality is Alex Morgan?"
    )

    assert (
        result.bridge_entity
        == "Alex Morgan"
    )

    assert fake.last_prompt is not None

    assert (
        "Selected bridge entity:\n"
        "Alex Morgan"
        in fake.last_prompt
    )

    assert (
        "Do not choose, replace, merge, "
        "or invent another"
        in fake.last_prompt
    )

    print(
        "PASS: valid constrained query was generated."
    )

    # --------------------------------------------------------
    # 2. Duplicate source passages are deduplicated
    # --------------------------------------------------------

    assert (
        fake.last_prompt.count(
            "Source rank: 1"
        )
        == 1
    )

    print(
        "PASS: duplicate candidate source passages "
        "were deduplicated."
    )

    # --------------------------------------------------------
    # 3. Optional outer JSON fence accepted
    # --------------------------------------------------------

    fenced_fake = FakeGenerator(
        '```json\n'
        '{"query": "Where was Alex Morgan born?"}\n'
        '```'
    )

    fenced_formulator = (
        ConstrainedRelationQueryGenerator(
            generator=fenced_fake
        )
    )

    fenced_result = (
        fenced_formulator.generate(
            question=question,
            selected_candidate=alex,
            candidates=candidates,
        )
    )

    assert (
        fenced_result.query
        == "Where was Alex Morgan born?"
    )

    print(
        "PASS: optional outer JSON code fence "
        "was accepted."
    )

    # --------------------------------------------------------
    # 4. Query omitting selected bridge is rejected
    # --------------------------------------------------------

    missing_bridge_fake = FakeGenerator(
        '{"query": "What nationality was '
        'the defending champion?"}'
    )

    missing_bridge_formulator = (
        ConstrainedRelationQueryGenerator(
            generator=missing_bridge_fake
        )
    )

    try:
        missing_bridge_formulator.generate(
            question=question,
            selected_candidate=alex,
            candidates=candidates,
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Query without selected bridge "
            "should have been rejected."
        )

    print(
        "PASS: query omitting the selected bridge "
        "was rejected."
    )

    # --------------------------------------------------------
    # 5. Multiple/free-form JSON fields rejected
    # --------------------------------------------------------

    extra_field_fake = FakeGenerator(
        '{"query": "What nationality is Alex Morgan?", '
        '"answer": "unknown"}'
    )

    extra_field_formulator = (
        ConstrainedRelationQueryGenerator(
            generator=extra_field_fake
        )
    )

    try:
        extra_field_formulator.generate(
            question=question,
            selected_candidate=alex,
            candidates=candidates,
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Extra JSON fields should "
            "have been rejected."
        )

    print(
        "PASS: extra/free-form JSON fields "
        "were rejected."
    )

    # --------------------------------------------------------
    # 6. Empty query rejected
    # --------------------------------------------------------

    empty_query_fake = FakeGenerator(
        '{"query": "   "}'
    )

    empty_query_formulator = (
        ConstrainedRelationQueryGenerator(
            generator=empty_query_fake
        )
    )

    try:
        empty_query_formulator.generate(
            question=question,
            selected_candidate=alex,
            candidates=candidates,
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Empty query should have been rejected."
        )

    print(
        "PASS: empty query was rejected."
    )

    # --------------------------------------------------------
    # 7. Selected candidate must belong to supplied pool
    # --------------------------------------------------------

    outside_candidate = make_candidate(
        entity="Jordan Lee",
        source_rank=3,
        source_chunk_id="chunk_003",
        source_title="Outside Source",
        source_text="Jordan Lee appears elsewhere.",
    )

    membership_fake = FakeGenerator(
        '{"query": "Who is Jordan Lee?"}'
    )

    membership_formulator = (
        ConstrainedRelationQueryGenerator(
            generator=membership_fake
        )
    )

    try:
        membership_formulator.generate(
            question=question,
            selected_candidate=outside_candidate,
            candidates=candidates,
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Selected candidate outside the "
            "candidate pool should be rejected."
        )

    print(
        "PASS: selected bridge must belong to "
        "the supplied candidate pool."
    )

    # --------------------------------------------------------
    # 8. Empty candidate pool rejected
    # --------------------------------------------------------

    empty_pool_fake = FakeGenerator(
        '{"query": "What nationality is Alex Morgan?"}'
    )

    empty_pool_formulator = (
        ConstrainedRelationQueryGenerator(
            generator=empty_pool_fake
        )
    )

    try:
        empty_pool_formulator.generate(
            question=question,
            selected_candidate=alex,
            candidates=(),
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Empty candidate pool should "
            "have been rejected."
        )

    print(
        "PASS: empty candidate pool was rejected."
    )

    print(
        "\nPHASE 10.3H-A UNIT TEST PASSED."
    )


if __name__ == "__main__":
    main()