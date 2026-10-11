from __future__ import annotations

from dataclasses import dataclass

from src.recovery.states import (
    RecoveryAction,
    RecoveryRequest,
    RecoveryResult,
    RecoveryStatus,
)


@dataclass(frozen=True)
class MultiQueryRecoveryConfig:
    """
    Configuration for deterministic evidence-conditioned
    multi-query recovery.
    """

    max_bridge_titles: int = 3
    source_result_depth: int = 5

    def __post_init__(self) -> None:

        if self.max_bridge_titles <= 0:
            raise ValueError(
                "max_bridge_titles must be positive."
            )

        if self.source_result_depth <= 0:
            raise ValueError(
                "source_result_depth must be positive."
            )


class MultiQueryRecovery:
    """
    Construct additional retrieval queries from document
    titles observed in the initial retrieved evidence.

    This stage performs query construction only.

    Retrieval/fusion of the generated queries will be added
    after the construction behavior is validated.
    """

    def __init__(
        self,
        config: MultiQueryRecoveryConfig | None = None,
    ) -> None:

        self.config = (
            config
            or MultiQueryRecoveryConfig()
        )

    @staticmethod
    def _normalize_title(
        title: str,
    ) -> str:

        return " ".join(
            str(title).split()
        ).strip()

    def _extract_bridge_titles(
        self,
        request: RecoveryRequest,
    ) -> tuple[str, ...]:

        titles: list[str] = []

        seen: set[str] = set()

        source_results = (
            request.original_results[
                : self.config.source_result_depth
            ]
        )

        for result in source_results:

            raw_title = (
                result.chunk.metadata.get(
                    "title",
                    "",
                )
            )

            title = self._normalize_title(
                raw_title
            )

            if not title:
                continue

            normalized_key = (
                title.casefold()
            )

            if normalized_key in seen:
                continue

            seen.add(
                normalized_key
            )

            titles.append(
                title
            )

            if (
                len(titles)
                >= self.config.max_bridge_titles
            ):
                break

        return tuple(
            titles
        )

    @staticmethod
    def _build_expanded_query(
        original_query: str,
        bridge_title: str,
    ) -> str:

        original_query = (
            " ".join(
                original_query.split()
            ).strip()
        )

        bridge_title = (
            " ".join(
                bridge_title.split()
            ).strip()
        )

        return (
            f"{original_query} "
            f"{bridge_title}"
        ).strip()

    def build_recovery_queries(
        self,
        request: RecoveryRequest,
    ) -> tuple[str, ...]:

        if (
            request.action
            != RecoveryAction.MULTI_QUERY_RETRIEVAL
        ):
            raise ValueError(
                "MultiQueryRecovery requires "
                "RecoveryAction.MULTI_QUERY_RETRIEVAL."
            )

        original_query = (
            " ".join(
                request.original_query.split()
            ).strip()
        )

        if not original_query:
            raise ValueError(
                "original_query cannot be empty."
            )

        bridge_titles = (
            self._extract_bridge_titles(
                request
            )
        )

        queries: list[str] = [
            original_query
        ]

        seen = {
            original_query.casefold()
        }

        for title in bridge_titles:

            expanded_query = (
                self._build_expanded_query(
                    original_query,
                    title,
                )
            )

            key = (
                expanded_query.casefold()
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            queries.append(
                expanded_query
            )

        return tuple(
            queries
        )

    def recover(
        self,
        request: RecoveryRequest,
    ) -> RecoveryResult:
        """
        Phase 10.3A validates deterministic query
        construction only.

        Candidate retrieval/fusion is intentionally deferred
        until the generated queries have been inspected.
        """

        try:

            recovery_queries = (
                self.build_recovery_queries(
                    request
                )
            )

            return RecoveryResult(
                action=request.action,
                status=RecoveryStatus.COMPLETED,
                original_query=(
                    request.original_query
                ),
                recovery_queries=(
                    recovery_queries
                ),
                original_results=(
                    request.original_results
                ),
                recovered_results=tuple(),
                error_message=None,
                metadata={
                    "strategy": (
                        "retrieved_title_multi_query"
                    ),
                    "max_bridge_titles": (
                        self.config
                        .max_bridge_titles
                    ),
                    "source_result_depth": (
                        self.config
                        .source_result_depth
                    ),
                    "query_count": len(
                        recovery_queries
                    ),
                    "retrieval_executed": False,
                },
            )

        except Exception as exc:

            return RecoveryResult(
                action=request.action,
                status=RecoveryStatus.FAILED,
                original_query=(
                    request.original_query
                ),
                recovery_queries=tuple(),
                original_results=(
                    request.original_results
                ),
                recovered_results=tuple(),
                error_message=str(exc),
                metadata={
                    "strategy": (
                        "retrieved_title_multi_query"
                    ),
                    "retrieval_executed": False,
                },
            )