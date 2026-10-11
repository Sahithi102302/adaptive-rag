from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from sentence_transformers import CrossEncoder

from src.recovery.entity_extraction import (
    ExtractedEntity,
)


@dataclass(frozen=True)
class BridgeSelectionConfig:
    """
    Configuration for question-conditioned bridge
    entity selection.

    The cross-encoder is used as an experimental
    relevance baseline. Its score must not be
    interpreted as a calibrated probability that an
    entity is the correct bridge.
    """

    model_name: str = (
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    batch_size: int = 32
    top_k: int = 3
    max_context_chars: int = 1200

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if self.top_k <= 0:
            raise ValueError(
                "top_k must be positive."
            )

        if self.max_context_chars <= 0:
            raise ValueError(
                "max_context_chars must be positive."
            )


@dataclass(frozen=True)
class BridgeCandidate:
    """
    Runtime-visible candidate bridge entity together
    with the source evidence from which it was
    extracted.

    ner_score describes confidence in named-entity
    extraction. It is preserved for analysis but is
    NOT used as bridge relevance.
    """

    entity: str
    entity_label: str
    ner_score: float

    source_rank: int
    source_chunk_id: str

    source_title: str
    source_text: str


@dataclass(frozen=True)
class ScoredBridgeCandidate:
    """
    One bridge candidate with its question-conditioned
    cross-encoder relevance score.
    """

    rank: int
    score: float
    candidate: BridgeCandidate


@dataclass(frozen=True)
class BridgeSelectionResult:
    """
    Complete output of one bridge-selection attempt.

    all_candidates preserves the runtime candidate
    pool.

    ranked_candidates contains at most config.top_k
    candidates ordered by cross-encoder relevance.
    """

    all_candidates: tuple[
        BridgeCandidate,
        ...
    ]

    ranked_candidates: tuple[
        ScoredBridgeCandidate,
        ...
    ]


class CrossEncoderBridgeSelector:
    """
    Experimental question-conditioned bridge selector.

    Pipeline:

        extracted entities
                +
        source retrieval results
                +
        original question
                |
                v
        candidate construction
                |
                v
        cross-encoder scoring
                |
                v
        ranked bridge candidates

    Important:
    - No gold evidence is used.
    - No gold answer is used.
    - No oracle failure label is used.
    - NER confidence is not treated as bridge
      relevance.
    - Cross-encoder scores are relevance scores, not
      calibrated bridge probabilities.
    """

    def __init__(
        self,
        config: BridgeSelectionConfig | None = None,
        model: Any | None = None,
    ) -> None:
        self.config = (
            config
            or BridgeSelectionConfig()
        )

        if model is None:
            self.model = CrossEncoder(
                self.config.model_name
            )
        else:
            self.model = model

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        return " ".join(
            str(text).split()
        ).strip()

    def _build_candidate_text(
        self,
        candidate: BridgeCandidate,
    ) -> str:
        """
        Construct the candidate-side text presented to
        the cross-encoder.

        The entity is made explicit, but its local
        source evidence is retained so the model can
        evaluate why the entity may matter to the
        question.
        """

        source_text = (
            self._normalize_text(
                candidate.source_text
            )
        )

        source_text = source_text[
            : self.config.max_context_chars
        ]

        return (
            f"Entity: {candidate.entity}\n"
            f"Source title: "
            f"{candidate.source_title}\n"
            f"Source passage: "
            f"{source_text}"
        )

    def build_candidates(
        self,
        entities: Sequence[ExtractedEntity],
        results: Sequence[Any],
    ) -> tuple[BridgeCandidate, ...]:
        """
        Resolve extracted entities back to the
        runtime-visible chunks from which they came.

        The entity extractor already records
        source_chunk_id, so no gold information is
        required for this join.
        """

        result_by_chunk_id = {
            result.chunk.id: result
            for result in results
        }

        candidates: list[
            BridgeCandidate
        ] = []

        for entity in entities:
            result = result_by_chunk_id.get(
                entity.source_chunk_id
            )

            if result is None:
                raise ValueError(
                    "Could not resolve extracted "
                    "entity source chunk: "
                    f"{entity.source_chunk_id!r}"
                )

            title = (
                result.chunk.metadata.get(
                    "title",
                    "",
                )
            )

            candidates.append(
                BridgeCandidate(
                    entity=(
                        self._normalize_text(
                            entity.text
                        )
                    ),
                    entity_label=(
                        self._normalize_text(
                            entity.label
                        )
                    ),
                    ner_score=float(
                        entity.score
                    ),
                    source_rank=int(
                        entity.source_rank
                    ),
                    source_chunk_id=(
                        entity.source_chunk_id
                    ),
                    source_title=(
                        self._normalize_text(
                            title
                        )
                    ),
                    source_text=(
                        self._normalize_text(
                            result.chunk.text
                        )
                    ),
                )
            )

        return tuple(
            candidates
        )

    def select(
        self,
        question: str,
        entities: Sequence[ExtractedEntity],
        results: Sequence[Any],
    ) -> BridgeSelectionResult:
        """
        Rank extracted bridge candidates against the
        original question.

        Runtime inputs:
            - original question
            - extracted entities
            - retrieved source passages

        Gold evidence and oracle information are not
        accepted by this interface.
        """

        question = self._normalize_text(
            question
        )

        if not question:
            raise ValueError(
                "question cannot be empty."
            )

        candidates = self.build_candidates(
            entities=entities,
            results=results,
        )

        if not candidates:
            return BridgeSelectionResult(
                all_candidates=(),
                ranked_candidates=(),
            )

        pairs: Sequence[
            tuple[str, str]
        ] = [
            (
                question,
                self._build_candidate_text(
                    candidate
                ),
            )
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )

        if len(scores) != len(
            candidates
        ):
            raise ValueError(
                "Cross-encoder returned a different "
                "number of scores than candidates."
            )

        scored = [
            (
                float(score),
                candidate,
            )
            for score, candidate in zip(
                scores,
                candidates,
            )
        ]

        # Deterministic ordering:
        #
        # 1. higher CE relevance
        # 2. earlier source retrieval rank
        # 3. entity text
        # 4. source chunk ID
        #
        # NER confidence is deliberately NOT used as
        # a bridge-selection tie-breaker.
        scored.sort(
            key=lambda item: (
                -item[0],
                item[1].source_rank,
                item[1].entity.casefold(),
                item[1].source_chunk_id,
            )
        )

        actual_k = min(
            self.config.top_k,
            len(scored),
        )

        ranked_candidates: list[
            ScoredBridgeCandidate
        ] = []

        for rank, (
            score,
            candidate,
        ) in enumerate(
            scored[:actual_k],
            start=1,
        ):
            ranked_candidates.append(
                ScoredBridgeCandidate(
                    rank=rank,
                    score=score,
                    candidate=candidate,
                )
            )

        return BridgeSelectionResult(
            all_candidates=candidates,
            ranked_candidates=tuple(
                ranked_candidates
            ),
        )