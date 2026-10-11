from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, Sequence

from src.recovery.bridge_selection import (
    BridgeCandidate,
)


class TextGenerator(Protocol):
    """
    Minimal interface required by the constrained
    relation/query formulator.
    """

    def generate(
        self,
        prompt: str,
    ) -> str:
        ...


@dataclass(frozen=True)
class RelationQueryConfig:
    """
    Configuration for constrained second-hop
    query formulation.
    """

    max_passage_chars: int = 1200

    def __post_init__(self) -> None:
        if self.max_passage_chars <= 0:
            raise ValueError(
                "max_passage_chars must be positive."
            )


@dataclass(frozen=True)
class RelationQueryResult:
    """
    Result of constrained second-hop query
    formulation.
    """

    query: str
    bridge_entity: str
    raw_output: str
    prompt: str


class ConstrainedRelationQueryGenerator:
    """
    Formulate one second-hop retrieval query for an
    already-selected bridge entity.

    This component does NOT choose the bridge entity.

    Inputs:
        - original question
        - first-hop runtime-visible evidence
        - one selected bridge candidate

    Output:
        - exactly one second-hop retrieval query

    Gold evidence, gold answers, oracle labels, and
    recovery-success information are intentionally
    outside this interface.
    """

    def __init__(
        self,
        generator: TextGenerator,
        config: RelationQueryConfig | None = None,
    ) -> None:
        self.generator = generator

        self.config = (
            config
            or RelationQueryConfig()
        )

    @staticmethod
    def _normalize_text(
        value: object,
    ) -> str:
        return " ".join(
            str(value).split()
        )

    def _build_evidence_block(
        self,
        candidates: Sequence[BridgeCandidate],
    ) -> str:
        """
        Build a deduplicated evidence block from the
        source passages represented by the supplied
        candidates.

        Candidate metadata is used only to reconstruct
        runtime-visible first-hop evidence.
        """

        unique_sources: dict[
            str,
            BridgeCandidate,
        ] = {}

        for candidate in candidates:
            if (
                candidate.source_chunk_id
                not in unique_sources
            ):
                unique_sources[
                    candidate.source_chunk_id
                ] = candidate

        ordered_sources = sorted(
            unique_sources.values(),
            key=lambda candidate: (
                candidate.source_rank,
                candidate.source_chunk_id,
            ),
        )

        blocks: list[str] = []

        for candidate in ordered_sources:
            title = self._normalize_text(
                candidate.source_title
            )

            text = self._normalize_text(
                candidate.source_text
            )

            text = text[
                : self.config.max_passage_chars
            ]

            blocks.append(
                "\n".join(
                    [
                        (
                            "Source rank: "
                            f"{candidate.source_rank}"
                        ),
                        f"Title: {title}",
                        f"Passage: {text}",
                    ]
                )
            )

        return "\n\n".join(
            blocks
        )

    def build_prompt(
        self,
        question: str,
        selected_candidate: BridgeCandidate,
        candidates: Sequence[BridgeCandidate],
    ) -> str:
        """
        Construct the constrained query-formulation
        prompt.

        The selected bridge entity is fixed input.
        The model is explicitly forbidden from
        replacing or reselecting it.
        """

        question = self._normalize_text(
            question
        )

        if not question:
            raise ValueError(
                "question cannot be empty."
            )

        if not candidates:
            raise ValueError(
                "candidates cannot be empty."
            )

        selected_entity = (
            self._normalize_text(
                selected_candidate.entity
            )
        )

        if not selected_entity:
            raise ValueError(
                "selected bridge entity "
                "cannot be empty."
            )

        if not any(
            candidate is selected_candidate
            or candidate == selected_candidate
            for candidate in candidates
        ):
            raise ValueError(
                "selected_candidate must belong "
                "to the supplied candidate pool."
            )

        evidence_block = (
            self._build_evidence_block(
                candidates
            )
        )

        return f"""
You are formulating ONE second-hop retrieval query.

The bridge entity has already been selected for you.
You MUST use that supplied bridge entity.
Do not choose, replace, merge, or invent another
bridge entity.

Your task:

1. Read the original question.
2. Read the first-hop retrieved evidence.
3. Determine what information is already resolved.
4. Determine what relation or information is still
   unresolved in the original question.
5. Formulate ONE concise search query that asks for
   the unresolved information about the supplied
   bridge entity.
6. The query must explicitly contain the supplied
   bridge entity.
7. Do not answer the original question.
8. Do not return explanations.
9. Do not return multiple queries.

Original question:
{question}

Selected bridge entity:
{selected_entity}

First-hop retrieved evidence:
{evidence_block}

Return exactly one JSON object in this format:

{{"query": "your second-hop retrieval query"}}
""".strip()

    @staticmethod
    def _strip_optional_code_fence(
        output: str,
    ) -> str:
        """
        Accept plain JSON or one optional outer
        Markdown code fence.
        """

        text = output.strip()

        if not text.startswith("```"):
            return text

        lines = text.splitlines()

        if len(lines) < 3:
            raise ValueError(
                "Malformed fenced JSON output."
            )

        opening = lines[0].strip()

        if opening not in {
            "```",
            "```json",
            "```JSON",
        }:
            raise ValueError(
                "Unsupported code fence."
            )

        if lines[-1].strip() != "```":
            raise ValueError(
                "Unclosed code fence."
            )

        return "\n".join(
            lines[1:-1]
        ).strip()

    @staticmethod
    def _parse_query(
        output: str,
        selected_entity: str,
    ) -> str:
        """
        Strictly parse one query from model output.

        Structural constraints:
        - valid JSON object
        - exactly one key: query
        - query must be a non-empty string
        - query must contain the supplied bridge entity
        """

        cleaned = (
            ConstrainedRelationQueryGenerator
            ._strip_optional_code_fence(
                output
            )
        )

        try:
            parsed = json.loads(
                cleaned
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Generator output is not valid JSON."
            ) from exc

        if not isinstance(
            parsed,
            dict,
        ):
            raise ValueError(
                "Generator output must be "
                "a JSON object."
            )

        if set(parsed.keys()) != {
            "query"
        }:
            raise ValueError(
                "Generator output must contain "
                "exactly one key: 'query'."
            )

        query = parsed["query"]

        if not isinstance(
            query,
            str,
        ):
            raise ValueError(
                "'query' must be a string."
            )

        query = " ".join(
            query.split()
        )

        if not query:
            raise ValueError(
                "'query' cannot be empty."
            )

        normalized_query = (
            query.casefold()
        )

        normalized_entity = (
            " ".join(
                selected_entity.split()
            ).casefold()
        )

        if (
            normalized_entity
            not in normalized_query
        ):
            raise ValueError(
                "Generated query must contain "
                "the supplied bridge entity."
            )

        return query

    def generate(
        self,
        question: str,
        selected_candidate: BridgeCandidate,
        candidates: Sequence[BridgeCandidate],
    ) -> RelationQueryResult:
        """
        Generate exactly one constrained second-hop
        retrieval query.
        """

        prompt = self.build_prompt(
            question=question,
            selected_candidate=selected_candidate,
            candidates=candidates,
        )

        raw_output = self.generator.generate(
            prompt
        )

        selected_entity = (
            self._normalize_text(
                selected_candidate.entity
            )
        )

        query = self._parse_query(
            output=raw_output,
            selected_entity=selected_entity,
        )

        return RelationQueryResult(
            query=query,
            bridge_entity=selected_entity,
            raw_output=raw_output,
            prompt=prompt,
        )