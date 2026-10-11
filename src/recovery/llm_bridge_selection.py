from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, Sequence

from src.recovery.bridge_selection import BridgeCandidate


class TextGenerator(Protocol):
    """
    Minimal text-generation interface.

    The bridge selector depends only on this protocol rather
    than on a specific local or hosted language model.
    """

    def generate(self, prompt: str) -> str:
        ...


@dataclass(frozen=True)
class LLMBridgeSelectionConfig:
    """
    Configuration for closed-candidate bridge selection.

    max_passage_chars:
        Maximum number of characters retained from each source
        passage when constructing the prompt.

    include_ner_metadata:
        Whether to expose the NER label in the candidate list.

        NER confidence is intentionally not included because
        previous experiments showed that NER confidence is not
        equivalent to bridge relevance.
    """

    max_passage_chars: int = 1200
    include_ner_metadata: bool = True

    def __post_init__(self) -> None:
        if self.max_passage_chars <= 0:
            raise ValueError(
                "max_passage_chars must be positive."
            )


@dataclass(frozen=True)
class LLMBridgeSelectionResult:
    """
    Result of a closed-candidate bridge-selection decision.

    selected_candidate:
        The exact BridgeCandidate selected from the supplied
        candidate pool.

    selected_index:
        One-based index used in the prompt.

    raw_output:
        Raw model output retained for diagnostics.

    prompt:
        Exact prompt supplied to the generator.
    """

    selected_candidate: BridgeCandidate
    selected_index: int
    raw_output: str
    prompt: str


class ClosedCandidateBridgeSelector:
    """
    Select exactly one bridge entity from a supplied candidate
    set.

    The language model is not allowed to invent entities.

    This component performs only bridge-entity selection.
    It does NOT:

        - formulate a second-hop query,
        - perform retrieval,
        - use gold evidence,
        - use gold answers,
        - use oracle failure labels,
        - determine whether recovery succeeded.

    Separating entity selection from relation/query formulation
    reduces the planning burden placed on the language model.
    """

    def __init__(
        self,
        generator: TextGenerator,
        config: LLMBridgeSelectionConfig | None = None,
    ) -> None:
        self.generator = generator
        self.config = (
            config
            if config is not None
            else LLMBridgeSelectionConfig()
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(
            str(text).strip().split()
        )

    def _build_evidence_block(
        self,
        candidates: Sequence[BridgeCandidate],
    ) -> str:
        """
        Build a deduplicated evidence block from candidate
        source passages.

        Multiple entities may originate from the same retrieved
        chunk. The passage is therefore shown only once.
        """

        evidence_parts: list[str] = []
        seen_chunks: set[str] = set()

        ordered_candidates = sorted(
            candidates,
            key=lambda candidate: (
                candidate.source_rank,
                candidate.source_chunk_id,
            ),
        )

        for candidate in ordered_candidates:
            if candidate.source_chunk_id in seen_chunks:
                continue

            seen_chunks.add(
                candidate.source_chunk_id
            )

            title = self._normalize_text(
                candidate.source_title
            )

            passage = self._normalize_text(
                candidate.source_text
            )

            passage = passage[
                : self.config.max_passage_chars
            ]

            evidence_parts.append(
                (
                    f"[Retrieved passage rank "
                    f"{candidate.source_rank}]\n"
                    f"Title: {title}\n"
                    f"Passage: {passage}"
                )
            )

        return "\n\n".join(
            evidence_parts
        )

    def _build_candidate_block(
        self,
        candidates: Sequence[BridgeCandidate],
    ) -> str:
        parts: list[str] = []

        for index, candidate in enumerate(
            candidates,
            start=1,
        ):
            entity = self._normalize_text(
                candidate.entity
            )

            if self.config.include_ner_metadata:
                parts.append(
                    (
                        f"[{index}] {entity} "
                        f"(NER label: "
                        f"{candidate.entity_label}, "
                        f"source passage rank: "
                        f"{candidate.source_rank})"
                    )
                )
            else:
                parts.append(
                    (
                        f"[{index}] {entity} "
                        f"(source passage rank: "
                        f"{candidate.source_rank})"
                    )
                )

        return "\n".join(parts)

    def build_prompt(
        self,
        question: str,
        candidates: Sequence[BridgeCandidate],
    ) -> str:
        if not candidates:
            raise ValueError(
                "Cannot build a bridge-selection prompt "
                "without candidates."
            )

        normalized_question = (
            self._normalize_text(question)
        )

        evidence_block = (
            self._build_evidence_block(
                candidates
            )
        )

        candidate_block = (
            self._build_candidate_block(
                candidates
            )
        )

        return f"""You are selecting a bridge entity for multi-hop retrieval.

Your task is ONLY to choose which supplied candidate entity is most useful for resolving the part of the original question that is still unanswered by the retrieved passages.

Instructions:
1. Read the original question.
2. Read the retrieved passages.
3. Determine what information is already resolved by those passages.
4. Determine what relation or information is still needed to answer the original question.
5. Select the supplied candidate entity that should be followed in a second retrieval step to obtain that missing information.
6. You MUST choose exactly one candidate from the supplied candidate list.
7. You MUST NOT invent, rewrite, merge, or add an entity.
8. Do NOT generate a retrieval query.
9. Do NOT answer the original question.

Return strict JSON only in exactly this format:
{{"selected_index": 1}}

ORIGINAL QUESTION:
{normalized_question}

RETRIEVED PASSAGES:
{evidence_block}

CANDIDATE ENTITIES:
{candidate_block}
"""

    @staticmethod
    def _strip_optional_code_fence(
        text: str,
    ) -> str:
        """
        Allow one optional outer Markdown code fence because
        small instruction-tuned models sometimes wrap otherwise
        valid JSON in ```json ... ```.

        Content inside the fence must still parse as strict JSON.
        """

        stripped = text.strip()

        if not stripped.startswith("```"):
            return stripped

        lines = stripped.splitlines()

        if len(lines) < 3:
            return stripped

        if not lines[-1].strip().startswith("```"):
            return stripped

        first_line = lines[0].strip()

        if first_line not in {
            "```",
            "```json",
            "```JSON",
        }:
            return stripped

        return "\n".join(
            lines[1:-1]
        ).strip()

    def _parse_selection(
        self,
        raw_output: str,
        candidate_count: int,
    ) -> int:
        cleaned = (
            self._strip_optional_code_fence(
                raw_output
            )
        )

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Bridge selector returned invalid JSON: "
                f"{raw_output!r}"
            ) from exc

        if not isinstance(parsed, dict):
            raise ValueError(
                "Bridge selector output must be a JSON object."
            )

        if set(parsed.keys()) != {
            "selected_index"
        }:
            raise ValueError(
                "Bridge selector JSON must contain exactly "
                "one key: 'selected_index'."
            )

        selected_index = parsed[
            "selected_index"
        ]

        if (
            isinstance(selected_index, bool)
            or not isinstance(
                selected_index,
                int,
            )
        ):
            raise ValueError(
                "'selected_index' must be an integer."
            )

        if not (
            1
            <= selected_index
            <= candidate_count
        ):
            raise ValueError(
                "'selected_index' is outside the supplied "
                "candidate range."
            )

        return selected_index

    def select(
        self,
        question: str,
        candidates: Sequence[BridgeCandidate],
    ) -> LLMBridgeSelectionResult:
        """
        Ask the generator to select exactly one supplied bridge
        candidate.

        The returned BridgeCandidate is resolved by index from
        the original candidate sequence. Therefore a model
        cannot introduce a new entity through its output.
        """

        if not candidates:
            raise ValueError(
                "Cannot select a bridge entity from an "
                "empty candidate pool."
            )

        prompt = self.build_prompt(
            question=question,
            candidates=candidates,
        )

        raw_output = self.generator.generate(
            prompt
        )

        selected_index = (
            self._parse_selection(
                raw_output=raw_output,
                candidate_count=len(
                    candidates
                ),
            )
        )

        selected_candidate = candidates[
            selected_index - 1
        ]

        return LLMBridgeSelectionResult(
            selected_candidate=(
                selected_candidate
            ),
            selected_index=selected_index,
            raw_output=raw_output,
            prompt=prompt,
        )