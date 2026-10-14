
"""Phase 11.4G-12H-9: grammar-paired title extractor v6.

Development-only title-boundary diagnostic.

- Pair each recognized prefix with its valid suffix grammar.
- Enumerate plausible boundaries instead of taking the first.
- Abstain when different title spans are plausible.
- Expose candidates for independent evidence validation.
- Do not use extraction to authorize fallback acceptance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class TitleSpanStatus(str, Enum):
    FOUND = "found"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class TitleSpanResult:
    status: TitleSpanStatus
    title: str | None
    start: int | None
    end: int | None
    reason: str

    @property
    def found(self) -> bool:
        return self.status == TitleSpanStatus.FOUND


@dataclass(frozen=True)
class TitleBoundaryCandidate:
    title: str
    start: int
    end: int
    grammar: str


@dataclass(frozen=True)
class Grammar:
    name: str
    prefix: re.Pattern[str]
    suffixes: tuple[re.Pattern[str], ...]


def compile_pattern(expression: str) -> re.Pattern[str]:
    return re.compile(expression, re.IGNORECASE)


ATTENDANCE_SUFFIXES = (
    compile_pattern(
        r"\s+(?:attend|attended)\s*\?\s*$"
    ),
    compile_pattern(
        r"\s+(?:never|not)\s+(?:attend|attended)\s*\?\s*$"
    ),
    compile_pattern(
        r"\s+(?:attend|attended)\s+(?:first|last)\s*\?\s*$"
    ),
    compile_pattern(
        r"\s+(?:attend|attended)\s+"
        r"(?:in|during|since|until|by|around)\s+"
        r"(?:18|19|20|21)\d{2}\s*\?\s*$"
    ),
    compile_pattern(
        r"\s+(?:attend|attended)\s+"
        r"(?:before|after)\s+.+\?\s*$"
    ),
)


GRAMMARS = (
    Grammar(
        name="author_university_attendance",
        prefix=compile_pattern(
            r"^\s*Which university did the "
            r"(?:author|writer|novelist) of\s+"
        ),
        suffixes=ATTENDANCE_SUFFIXES,
    ),
    Grammar(
        name="author_college_education",
        prefix=compile_pattern(
            r"^\s*At what college was the "
            r"(?:author|writer|novelist) of\s+"
        ),
        suffixes=(
            compile_pattern(
                r"\s+educated\s*\?\s*$"
            ),
        ),
    ),
    Grammar(
        name="authored_person_college",
        prefix=compile_pattern(
            r"^\s*Where did the person who authored\s+"
        ),
        suffixes=(
            compile_pattern(
                r"\s+go to college\s*\?\s*$"
            ),
        ),
    ),
)


class ConservativeTitleSpanExtractor:
    name = "conservative_title_span_extractor_v6"

    def enumerate_candidates(
        self,
        question: str,
    ) -> tuple[TitleBoundaryCandidate, ...]:
        """Return plausible title spans without selecting a winner."""

        if not isinstance(question, str) or not question.strip():
            return ()

        matched_grammars = []

        for grammar in GRAMMARS:
            prefix_match = grammar.prefix.match(question)

            if prefix_match is not None:
                matched_grammars.append(
                    (grammar, prefix_match)
                )

        if len(matched_grammars) != 1:
            return ()

        grammar, prefix_match = matched_grammars[0]
        title_start = prefix_match.end()

        candidates: set[TitleBoundaryCandidate] = set()

        for suffix_pattern in grammar.suffixes:
            for suffix_match in suffix_pattern.finditer(question):
                title_end = suffix_match.start()

                if title_end <= title_start:
                    continue

                title = question[
                    title_start:title_end
                ].strip()

                if not title or not re.search(r"[A-Za-z]", title):
                    continue

                start = title_start
                end = start + len(title)

                if question[start:end] != title:
                    continue

                candidates.add(
                    TitleBoundaryCandidate(
                        title=title,
                        start=start,
                        end=end,
                        grammar=grammar.name,
                    )
                )

        return tuple(
            sorted(
                candidates,
                key=lambda item: (
                    item.start,
                    item.end,
                    item.title,
                ),
            )
        )

    def extract(self, question: str) -> TitleSpanResult:
        if not isinstance(question, str) or not question.strip():
            return self._unresolved("empty_question")

        matched_grammars = [
            grammar
            for grammar in GRAMMARS
            if grammar.prefix.match(question) is not None
        ]

        if len(matched_grammars) != 1:
            return self._unresolved(
                "unsupported_or_ambiguous_prefix"
            )

        candidates = self.enumerate_candidates(question)

        if not candidates:
            return self._unresolved(
                "no_title_boundary_candidate"
            )

        if len(candidates) > 1:
            return self._unresolved(
                "ambiguous_multiple_title_boundaries"
            )

        candidate = candidates[0]

        return TitleSpanResult(
            status=TitleSpanStatus.FOUND,
            title=candidate.title,
            start=candidate.start,
            end=candidate.end,
            reason=f"unique_boundary:{candidate.grammar}",
        )

    @staticmethod
    def _unresolved(reason: str) -> TitleSpanResult:
        return TitleSpanResult(
            status=TitleSpanStatus.UNRESOLVED,
            title=None,
            start=None,
            end=None,
            reason=reason,
        )
