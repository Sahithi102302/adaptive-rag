from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from src.retrieval.reranker import (
    RerankedRetrievalResult,
)


class EvidenceSource(str, Enum):
    FIRST_HOP = "first_hop"
    SECOND_HOP = "second_hop"
    BOTH = "both"


@dataclass(frozen=True)
class EvidenceSelectionConfig:
    final_k: int = 10
    reserved_first_hop: int = 5
    reserved_second_hop: int = 5

    def __post_init__(self) -> None:
        if self.final_k <= 0:
            raise ValueError(
                "final_k must be greater than 0."
            )

        if self.reserved_first_hop < 0:
            raise ValueError(
                "reserved_first_hop cannot be negative."
            )

        if self.reserved_second_hop < 0:
            raise ValueError(
                "reserved_second_hop cannot be negative."
            )

        if (
            self.reserved_first_hop
            + self.reserved_second_hop
            > self.final_k
        ):
            raise ValueError(
                "Combined reserved slots cannot "
                "exceed final_k."
            )


@dataclass(frozen=True)
class SelectedEvidence:
    """
    A selected chunk with its retrieval provenance.

    Scores from different retrieval queries are deliberately
    not compared directly.
    """

    rank: int
    chunk_id: str
    chunk: object
    source: EvidenceSource

    first_hop_rank: int | None
    second_hop_rank: int | None


@dataclass(frozen=True)
class EvidenceSelectionResult:
    original_results: tuple[
        RerankedRetrievalResult, ...
    ]

    recovered_results: tuple[
        RerankedRetrievalResult, ...
    ]

    selected_results: tuple[
        SelectedEvidence, ...
    ]


class EvidencePreservingSelector:
    """
    Deterministic, hop-balanced selection baseline.

    Policy:
      1. Deduplicate each hop by chunk ID.
      2. Reserve first-hop and second-hop slots.
      3. Deduplicate across hops.
      4. Fill remaining slots in alternating hop order.
      5. Preserve provenance and original per-hop ranks.

    This is a candidate selection policy, not an evidence
    sufficiency verifier.
    """

    def __init__(
        self,
        config: EvidenceSelectionConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else EvidenceSelectionConfig()
        )

    @staticmethod
    def _deduplicate(
        results: Sequence[RerankedRetrievalResult],
    ) -> list[RerankedRetrievalResult]:

        seen: set[str] = set()
        unique: list[RerankedRetrievalResult] = []

        for result in results:
            chunk_id = result.chunk.id

            if chunk_id in seen:
                continue

            seen.add(chunk_id)
            unique.append(result)

        return unique

    def select(
        self,
        original_results: Sequence[
            RerankedRetrievalResult
        ],
        recovered_results: Sequence[
            RerankedRetrievalResult
        ],
    ) -> EvidenceSelectionResult:

        first = self._deduplicate(
            original_results
        )

        second = self._deduplicate(
            recovered_results
        )

        first_by_id = {
            result.chunk.id: result
            for result in first
        }

        second_by_id = {
            result.chunk.id: result
            for result in second
        }

        selected_ids: list[str] = []
        selected_set: set[str] = set()

        def add(chunk_id: str) -> None:
            if len(selected_ids) >= self.config.final_k:
                return

            if chunk_id in selected_set:
                return

            selected_ids.append(chunk_id)
            selected_set.add(chunk_id)

        # Reserve first-hop slots.
        for result in first[
            :self.config.reserved_first_hop
        ]:
            add(result.chunk.id)

        # Reserve second-hop slots.
        for result in second[
            :self.config.reserved_second_hop
        ]:
            add(result.chunk.id)

        # Fill remaining capacity, alternating between hops.
        first_remaining = [
            result.chunk.id
            for result in first
            if result.chunk.id not in selected_set
        ]

        second_remaining = [
            result.chunk.id
            for result in second
            if result.chunk.id not in selected_set
        ]

        first_index = 0
        second_index = 0

        while (
            len(selected_ids) < self.config.final_k
            and (
                first_index < len(first_remaining)
                or second_index < len(second_remaining)
            )
        ):
            if first_index < len(first_remaining):
                add(first_remaining[first_index])
                first_index += 1

            if (
                len(selected_ids) < self.config.final_k
                and second_index < len(second_remaining)
            ):
                add(second_remaining[second_index])
                second_index += 1

        selected: list[SelectedEvidence] = []

        for rank, chunk_id in enumerate(
            selected_ids,
            start=1,
        ):
            first_result = first_by_id.get(
                chunk_id
            )

            second_result = second_by_id.get(
                chunk_id
            )

            if (
                first_result is not None
                and second_result is not None
            ):
                source = EvidenceSource.BOTH

            elif first_result is not None:
                source = EvidenceSource.FIRST_HOP

            else:
                source = EvidenceSource.SECOND_HOP

            reference = (
                first_result
                if first_result is not None
                else second_result
            )

            assert reference is not None

            selected.append(
                SelectedEvidence(
                    rank=rank,
                    chunk_id=chunk_id,
                    chunk=reference.chunk,
                    source=source,
                    first_hop_rank=(
                        first_result.rank
                        if first_result is not None
                        else None
                    ),
                    second_hop_rank=(
                        second_result.rank
                        if second_result is not None
                        else None
                    ),
                )
            )

        return EvidenceSelectionResult(
            original_results=tuple(
                original_results
            ),
            recovered_results=tuple(
                recovered_results
            ),
            selected_results=tuple(
                selected
            ),
        )