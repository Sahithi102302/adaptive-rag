from __future__ import annotations

import numpy as np

from src.chunking.chunk import Chunk
from src.recovery.bridge_selection import (
    BridgeSelectionConfig,
    CrossEncoderBridgeSelector,
)
from src.recovery.entity_extraction import (
    ExtractedEntity,
)
from src.retrieval.reranker import (
    RerankedRetrievalResult,
)


class FakeCrossEncoder:
    """
    Deterministic fake model for testing bridge
    selection mechanics without loading a real
    transformer.
    """

    def predict(
        self,
        pairs,
        batch_size,
        show_progress_bar,
    ):
        scores = []

        for question, candidate_text in pairs:

            assert (
                "nationality"
                in question.lower()
            )

            if (
                "Entity: Lin Dan"
                in candidate_text
            ):
                scores.append(
                    0.90
                )

            elif (
                "Entity: Wang Yihan"
                in candidate_text
            ):
                scores.append(
                    0.30
                )

            elif (
                "Entity: Chen Jin"
                in candidate_text
            ):
                scores.append(
                    0.20
                )

            else:
                scores.append(
                    0.0
                )

        return np.asarray(
            scores,
            dtype=np.float32,
        )


def make_result(
    rank: int,
    chunk_id: str,
    title: str,
    text: str,
) -> RerankedRetrievalResult:

    chunk = Chunk(
        id=chunk_id,
        document_id=chunk_id,
        chunk_index=0,
        text=text,
        start_char=0,
        end_char=len(text),
        metadata={
            "title": title,
        },
    )

    return RerankedRetrievalResult(
        rank=rank,
        score=1.0,
        chunk=chunk,
    )


def main() -> None:

    print(
        "=" * 80
    )

    print(
        "PHASE 10.3F-A — BRIDGE "
        "SELECTION UNIT TEST"
    )

    print(
        "=" * 80
    )

    question = (
        "What was the nationality of the "
        "defending champion?"
    )

    results = [
        make_result(
            rank=1,
            chunk_id="chunk-bwf-2013",
            title=(
                "2013 BWF World Championships "
                "– Men's singles"
            ),
            text=(
                "Lin Dan was the defending "
                "champion."
            ),
        ),
        make_result(
            rank=2,
            chunk_id="chunk-bwf-women",
            title=(
                "2013 BWF World Championships "
                "– Women's singles"
            ),
            text=(
                "Wang Yihan was the defending "
                "champion."
            ),
        ),
        make_result(
            rank=3,
            chunk_id="chunk-bwf-2011",
            title=(
                "2011 BWF World Championships "
                "– Men's singles"
            ),
            text=(
                "Chen Jin was the defending "
                "champion."
            ),
        ),
    ]

    entities = (
        ExtractedEntity(
            text="Lin Dan",
            label="PER",
            score=0.99,
            source_rank=1,
            source_chunk_id=(
                "chunk-bwf-2013"
            ),
        ),
        ExtractedEntity(
            text="Wang Yihan",
            label="PER",
            score=0.999,
            source_rank=2,
            source_chunk_id=(
                "chunk-bwf-women"
            ),
        ),
        ExtractedEntity(
            text="Chen Jin",
            label="PER",
            score=0.9999,
            source_rank=3,
            source_chunk_id=(
                "chunk-bwf-2011"
            ),
        ),
    )

    selector = CrossEncoderBridgeSelector(
        config=BridgeSelectionConfig(
            top_k=2,
        ),
        model=FakeCrossEncoder(),
    )

    selection = selector.select(
        question=question,
        entities=entities,
        results=results,
    )

    print(
        "\nAll candidates:"
    )

    for candidate in (
        selection.all_candidates
    ):
        print(
            f"  entity={candidate.entity!r} "
            f"ner_score="
            f"{candidate.ner_score:.4f} "
            f"source_rank="
            f"{candidate.source_rank}"
        )

    print(
        "\nRanked candidates:"
    )

    for scored in (
        selection.ranked_candidates
    ):
        print(
            f"  rank={scored.rank} "
            f"entity="
            f"{scored.candidate.entity!r} "
            f"bridge_score="
            f"{scored.score:.4f}"
        )

    assert (
        len(selection.all_candidates)
        == 3
    )

    print(
        "\nPASS: all extracted candidates "
        "were preserved."
    )

    assert (
        len(selection.ranked_candidates)
        == 2
    )

    print(
        "PASS: top-k truncation works."
    )

    assert (
        selection
        .ranked_candidates[0]
        .candidate.entity
        == "Lin Dan"
    )

    print(
        "PASS: highest bridge-relevance "
        "candidate ranked first."
    )

    assert (
        selection
        .ranked_candidates[1]
        .candidate.entity
        == "Wang Yihan"
    )

    print(
        "PASS: candidate ranking is "
        "deterministic."
    )

    assert (
        selection
        .ranked_candidates[0]
        .candidate.source_chunk_id
        == "chunk-bwf-2013"
    )

    assert (
        selection
        .ranked_candidates[0]
        .candidate.source_title
        == (
            "2013 BWF World Championships "
            "– Men's singles"
        )
    )

    print(
        "PASS: source metadata was "
        "preserved."
    )

    assert (
        selection
        .ranked_candidates[0]
        .candidate.ner_score
        < selection
        .ranked_candidates[1]
        .candidate.ner_score
    )

    print(
        "PASS: NER confidence did not "
        "control bridge ranking."
    )

    empty_selection = selector.select(
        question=question,
        entities=(),
        results=results,
    )

    assert (
        empty_selection.all_candidates
        == ()
    )

    assert (
        empty_selection.ranked_candidates
        == ()
    )

    print(
        "PASS: empty candidate pools are "
        "handled safely."
    )

    print(
        "\nPHASE 10.3F-A UNIT TEST PASSED."
    )


if __name__ == "__main__":
    main()