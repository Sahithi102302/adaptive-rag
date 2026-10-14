
"""Phase 11.4G-12H-10C: question structure diagnostic.

Classifies a limited set of observable question structures.

This is NOT a semantic entailment verifier.
It does not extract a complete reasoning graph.
It does not authorize decomposition acceptance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class QuestionStructure(str, Enum):
    EXPLICIT_COMPARISON = "explicit_comparison"
    POSSIBLE_BRIDGE = "possible_bridge"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class QuestionStructureResult:
    structure: QuestionStructure
    constraints: tuple[str, ...]
    reason: str


class QuestionStructureDiagnostic:
    name = "question_structure_diagnostic_v1"

    _COMPARISON_PATTERNS = (
    re.compile(r"^\s*(?:are|were)\b.+\b(?:same|both)\b", re.I),
    re.compile(r"^\s*(?:who|which)\b.+\b(?:older|younger)\b", re.I),
    re.compile(r"\b(?:higher|lower|greater|smaller)\b.+\bthan\b", re.I),

    # Candidate-alternative comparison.
    # Deliberately excludes questions beginning with Is/Are.
    re.compile(
        r"^\s*(?:who|which)\b.+\bor\b.+\?",
        re.I,
    ),
)

    _BRIDGE_SIGNALS = (
        re.compile(r"\b(?:whose|where|who)\b", re.I),
        re.compile(r"\b(?:director|author|father|mother|singer)\s+of\b", re.I),
        re.compile(r"\b(?:portrayed|recruited|founded|voiced)\b", re.I),
        re.compile(r"\b(?:that|which)\s+(?:was|were|has|had|is)\b", re.I),
    )

    _CONSTRAINT_PATTERNS = {
        "negation": re.compile(
            r"\b(?:not|never|neither|without)\b", re.I
        ),
        "temporal": re.compile(
            r"\b(?:before|after|during|since|until|formerly)\b"
            r"|\b(?:18|19|20|21)\d{2}\b",
            re.I,
        ),
        "superlative": re.compile(
            r"\b(?:first|last|oldest|youngest|largest|smallest)\b",
            re.I,
        ),
        "comparison": re.compile(
            r"\b(?:same|both|older|younger|higher|lower|greater)\b",
            re.I,
        ),
    }

    def analyze(self, question: str) -> QuestionStructureResult:
        if not isinstance(question, str) or not question.strip():
            return QuestionStructureResult(
                structure=QuestionStructure.UNRESOLVED,
                constraints=(),
                reason="invalid_question",
            )

        constraints = tuple(
            name
            for name, pattern in self._CONSTRAINT_PATTERNS.items()
            if pattern.search(question)
        )

        if any(
            pattern.search(question)
            for pattern in self._COMPARISON_PATTERNS
        ):
            return QuestionStructureResult(
                structure=QuestionStructure.EXPLICIT_COMPARISON,
                constraints=constraints,
                reason="comparison_surface_pattern",
            )

        if any(
            pattern.search(question)
            for pattern in self._BRIDGE_SIGNALS
        ):
            return QuestionStructureResult(
                structure=QuestionStructure.POSSIBLE_BRIDGE,
                constraints=constraints,
                reason="bridge_surface_signal",
            )

        return QuestionStructureResult(
            structure=QuestionStructure.UNRESOLVED,
            constraints=constraints,
            reason="insufficient_structure_evidence",
        )
