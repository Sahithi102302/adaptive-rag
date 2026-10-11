from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import pipeline


@dataclass(frozen=True)
class EntityExtractionConfig:
    """
    Configuration for content-derived named-entity extraction.

    max_entities:
        Final global safety cap across all inspected source
        passages.

    max_entities_per_source:
        Optional per-passage entity budget.

        None preserves the original global-budget behavior.

        When set, each retrieved source passage may contribute
        at most this many unique entities before extraction
        proceeds to the next passage. This prevents a noisy
        high-ranked passage from consuming the entire global
        candidate budget.
    """

    model_name: str = "dslim/bert-base-NER"
    source_result_depth: int = 3
    max_entities: int = 10
    max_entities_per_source: int | None = None
    aggregation_strategy: str = "simple"

    def __post_init__(self) -> None:
        if self.source_result_depth <= 0:
            raise ValueError(
                "source_result_depth must be positive."
            )

        if self.max_entities <= 0:
            raise ValueError(
                "max_entities must be positive."
            )

        if (
            self.max_entities_per_source is not None
            and self.max_entities_per_source <= 0
        ):
            raise ValueError(
                "max_entities_per_source must be "
                "positive when provided."
            )


@dataclass(frozen=True)
class ExtractedEntity:
    """
    A named entity extracted from runtime-visible retrieved
    evidence.

    source_rank and source_chunk_id preserve provenance so
    downstream recovery components can resolve the entity
    back to the passage from which it was extracted.
    """

    text: str
    label: str
    score: float
    source_rank: int
    source_chunk_id: str


class ContentEntityExtractor:
    """
    Extract named entities from retrieved passage content.

    This component uses only runtime-visible retrieved
    passages. It receives no gold evidence, supporting-fact
    annotations, gold document IDs, answers, or oracle failure
    labels.

    Entity extraction and bridge selection are intentionally
    separate responsibilities:

        retrieved passages
            -> named-entity extraction
            -> bridge candidate selection
            -> later recovery planning

    The extractor supports two candidate-budget modes:

    1. Original global-budget behavior
       max_entities_per_source=None

       Passages are processed in retrieval order until the
       global max_entities limit is reached.

    2. Passage-balanced behavior
       max_entities_per_source=N

       Each source passage can contribute at most N unique
       entities, while max_entities remains the final global
       safety cap.
    """

    def __init__(
        self,
        config: EntityExtractionConfig | None = None,
        ner_pipeline: Any | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else EntityExtractionConfig()
        )

        if ner_pipeline is not None:
            self._ner = ner_pipeline
        else:
            self._ner = pipeline(
                "ner",
                model=self.config.model_name,
                aggregation_strategy=(
                    self.config.aggregation_strategy
                ),
            )

    @staticmethod
    def _normalize_entity(
        text: str,
    ) -> str:
        """
        Normalize whitespace while preserving the entity text
        emitted by the NER model.

        We deliberately do not attempt aggressive WordPiece
        reconstruction here. Fragmentation remains observable
        so it can be evaluated separately from candidate-budget
        behavior.
        """

        return " ".join(
            str(text).strip().split()
        )

    @staticmethod
    def _entity_key(
        text: str,
    ) -> str:
        """
        Case-insensitive key used for deterministic
        deduplication.
        """

        return text.casefold()

    @staticmethod
    def _is_valid_entity(
        text: str,
    ) -> bool:
        """
        Reject empty/non-alphanumeric NER outputs.

        This intentionally remains a lightweight validity
        check rather than a semantic entity-quality filter.
        """

        if not text:
            return False

        return any(
            character.isalnum()
            for character in text
        )

    def extract_from_results(
        self,
        results,
    ) -> tuple[ExtractedEntity, ...]:
        """
        Extract unique named entities from the highest-ranked
        retrieved passages.

        Deduplication is case-insensitive. If the same entity
        occurs multiple times, the first occurrence from the
        highest-ranked source passage is retained.

        When max_entities_per_source is configured, each source
        passage receives its own entity budget. This prevents a
        noisy high-ranked passage from exhausting the complete
        candidate budget before later passages are inspected.

        max_entities always remains the final global safety cap.
        """

        extracted: list[
            ExtractedEntity
        ] = []

        seen: set[str] = set()

        source_results = results[
            : self.config.source_result_depth
        ]

        for result in source_results:
            source_entity_count = 0

            text = result.chunk.text

            predictions = self._ner(
                text
            )

            for prediction in predictions:
                raw_text = (
                    prediction.get("word")
                    or prediction.get(
                        "text",
                        "",
                    )
                )

                entity_text = (
                    self._normalize_entity(
                        raw_text
                    )
                )

                if not self._is_valid_entity(
                    entity_text
                ):
                    continue

                key = self._entity_key(
                    entity_text
                )

                if key in seen:
                    continue

                # --------------------------------------------
                # Passage-balanced candidate budget
                #
                # The check occurs after validation and
                # deduplication so invalid/duplicate entities
                # do not consume the source passage's budget.
                # --------------------------------------------

                if (
                    self.config.max_entities_per_source
                    is not None
                    and source_entity_count
                    >= self.config.max_entities_per_source
                ):
                    break

                seen.add(key)

                label = str(
                    prediction.get(
                        "entity_group",
                        prediction.get(
                            "entity",
                            "UNKNOWN",
                        ),
                    )
                )

                score = float(
                    prediction.get(
                        "score",
                        0.0,
                    )
                )

                extracted.append(
                    ExtractedEntity(
                        text=entity_text,
                        label=label,
                        score=score,
                        source_rank=(
                            result.rank
                        ),
                        source_chunk_id=(
                            result.chunk.id
                        ),
                    )
                )

                source_entity_count += 1

                # --------------------------------------------
                # Final global safety cap
                # --------------------------------------------

                if (
                    len(extracted)
                    >= self.config.max_entities
                ):
                    return tuple(
                        extracted
                    )

        return tuple(
            extracted
        )