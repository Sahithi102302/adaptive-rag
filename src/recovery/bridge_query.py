from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, Sequence


class TextGenerator(Protocol):
    """
    Minimal interface required by the bridge-query
    generator.

    Any backend that implements:

        generate(prompt: str) -> str

    can be used without coupling bridge-query logic
    to a particular LLM provider or model.
    """

    def generate(
        self,
        prompt: str,
    ) -> str:
        ...


@dataclass(frozen=True)
class BridgeQueryConfig:
    """
    Configuration for relation-aware bridge-query
    generation.
    """

    source_result_depth: int = 3
    max_queries: int = 3
    max_passage_chars: int = 1200

    def __post_init__(self) -> None:
        if self.source_result_depth <= 0:
            raise ValueError(
                "source_result_depth must be positive."
            )

        if self.max_queries <= 0:
            raise ValueError(
                "max_queries must be positive."
            )

        if self.max_passage_chars <= 0:
            raise ValueError(
                "max_passage_chars must be positive."
            )


@dataclass(frozen=True)
class BridgeQueryResult:
    """
    Immutable record of one bridge-query generation
    attempt.

    Keeping the raw output and exact prompt is useful
    for later error analysis and reproducibility.
    """

    queries: tuple[str, ...]
    raw_output: str
    prompt: str


class RelationAwareBridgeQueryGenerator:
    """
    Generate focused second-hop retrieval queries from:

        1. the original question
        2. initially retrieved evidence

    The generator must not receive:
        - gold supporting facts
        - gold document IDs
        - the gold answer
        - oracle failure labels

    This component only generates possible retrieval
    queries. It does not execute retrieval and does
    not determine whether recovery succeeded.
    """

    def __init__(
        self,
        text_generator: TextGenerator,
        config: BridgeQueryConfig | None = None,
    ) -> None:
        self.text_generator = text_generator

        self.config = (
            config
            or BridgeQueryConfig()
        )

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        """
        Collapse repeated whitespace while preserving
        textual content.
        """

        return " ".join(
            str(text).split()
        ).strip()

    def _build_evidence_block(
        self,
        results: Sequence[Any],
    ) -> str:
        """
        Convert the top retrieved results into the
        evidence block shown to the text generator.

        Only runtime-visible retrieval information is
        included.
        """

        blocks: list[str] = []

        source_results = results[
            : self.config.source_result_depth
        ]

        for result in source_results:
            title = self._normalize_text(
                result.chunk.metadata.get(
                    "title",
                    "",
                )
            )

            passage = self._normalize_text(
                result.chunk.text
            )

            passage = passage[
                : self.config.max_passage_chars
            ]

            blocks.append(
                (
                    f"[Retrieved passage "
                    f"rank={result.rank}]\n"
                    f"Title: {title}\n"
                    f"Text: {passage}"
                )
            )

        return "\n\n".join(
            blocks
        )

    def build_prompt(
        self,
        question: str,
        results: Sequence[Any],
    ) -> str:
        """
        Build a resolved/unresolved-aware bridge-query
        prompt.

        The model must distinguish information already
        established by retrieved evidence from the
        information still required to answer the
        original question.

        Only runtime-visible information is included.
        """

        question = self._normalize_text(
            question
        )

        if not question:
            raise ValueError(
                "question cannot be empty."
            )

        evidence_block = (
            self._build_evidence_block(
                results
            )
        )

        return f"""
You generate search queries for the next retrieval hop
in a multi-hop question-answering system.

Your task is NOT to answer the original question.

Use only:
1. the original question
2. the retrieved passages below

Before generating a query, reason about the retrieval
state:

Step 1 — Resolved information:
Determine what facts or relationships needed by the
original question are already explicitly established
by the retrieved passages.

Step 2 — Unresolved information:
Determine what information is still missing before
the original question can be answered.

Step 3 — Bridge query:
If the retrieved passages identify an entity needed
for the next hop, combine that resolved entity with
the unresolved relation to construct a focused search
query.

Important:
Do NOT generate a query whose answer is already
explicitly stated in the retrieved passages.

For example, if a question asks for an attribute of
a person and a retrieved passage already identifies
which person satisfies an earlier relation, do NOT
search for that earlier relation again. Search for
the missing attribute of the identified person.

Generate at most {self.config.max_queries} short,
focused search queries.

Rules:
- Do not answer the original question.
- Use only entities supported by the original
  question or retrieved passages.
- Do not invent bridge entities.
- Search for unresolved information, not information
  already explicitly stated in the passages.
- Prefer the entity that completes the unresolved
  relation in the original question.
- Avoid repeating the original question when the
  retrieved evidence has already resolved part of it.
- Avoid unrelated entities from retrieved passages.
- Return JSON only.
- Do not include your reasoning in the output.
- Use exactly this schema:

{{"queries": ["query 1", "query 2"]}}

Original question:
{question}

Retrieved passages:
{evidence_block}
""".strip()

    @staticmethod
    def _strip_markdown_code_fence(
        raw_output: str,
    ) -> str:
        """
        Remove one optional Markdown code fence around
        the model output.

        Example accepted form:

            ```json
            {"queries": ["example query"]}
            ```

        This is intentionally conservative.

        It does not:
        - search arbitrary prose for embedded JSON
        - repair malformed JSON
        - alter generated query content
        """

        cleaned_output = (
            raw_output.strip()
        )

        if not cleaned_output.startswith(
            "```"
        ):
            return cleaned_output

        lines = (
            cleaned_output.splitlines()
        )

        if not lines:
            return cleaned_output

        # Remove opening fence:
        #
        # ```
        #
        # or:
        #
        # ```json
        lines = lines[1:]

        # Remove the closing fence only when it is
        # the final non-whitespace line.
        if (
            lines
            and lines[-1].strip() == "```"
        ):
            lines = lines[:-1]

        return (
            "\n".join(
                lines
            ).strip()
        )

    def _parse_output(
        self,
        raw_output: str,
    ) -> tuple[str, ...]:
        """
        Parse structured bridge-query output.

        One optional Markdown code fence is treated
        only as a serialization wrapper.

        The content inside must still:
        - be valid JSON
        - be a JSON object
        - contain a "queries" list

        Malformed model output is not silently
        repaired.
        """

        cleaned_output = (
            self._strip_markdown_code_fence(
                raw_output
            )
        )

        try:
            parsed = json.loads(
                cleaned_output
            )

        except json.JSONDecodeError as exc:
            raise ValueError(
                "Bridge-query generator returned "
                "invalid JSON."
            ) from exc

        if not isinstance(
            parsed,
            dict,
        ):
            raise ValueError(
                "Bridge-query output must be "
                "a JSON object."
            )

        queries = parsed.get(
            "queries"
        )

        if not isinstance(
            queries,
            list,
        ):
            raise ValueError(
                "'queries' must be a JSON list."
            )

        normalized: list[str] = []
        seen: set[str] = set()

        for query in queries:
            if not isinstance(
                query,
                str,
            ):
                continue

            query = self._normalize_text(
                query
            )

            if not query:
                continue

            key = (
                query.casefold()
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            normalized.append(
                query
            )

            if (
                len(normalized)
                >= self.config.max_queries
            ):
                break

        return tuple(
            normalized
        )

    def generate(
        self,
        question: str,
        results: Sequence[Any],
    ) -> BridgeQueryResult:
        """
        Generate and parse bridge queries.

        Runtime inputs:
            - original question
            - retrieved results

        The raw model output and exact prompt are
        retained for later inspection.

        Gold evidence and oracle information are not
        accepted by this interface.
        """

        prompt = self.build_prompt(
            question=question,
            results=results,
        )

        raw_output = (
            self.text_generator.generate(
                prompt
            )
        )

        # Temporary experiment-time diagnostic.
        #
        # Keep this during Phase 10.3E so that the
        # exact local-model output remains visible
        # during controlled inspection.
        print(
            "\n[BridgeQueryGenerator raw output]"
        )

        print(
            repr(raw_output)
        )

        queries = (
            self._parse_output(
                raw_output
            )
        )

        return BridgeQueryResult(
            queries=queries,
            raw_output=raw_output,
            prompt=prompt,
        )