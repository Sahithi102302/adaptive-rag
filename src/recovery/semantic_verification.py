
"""Structured semantic verification contracts for AdaptiveRAG.

Phase 11.4A defines validated semantic assessment outputs.
It does not run a model, establish factual correctness, or accept recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SemanticSupport(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class SemanticFindingType(str, Enum):
    SUPPORT = "support"
    MISSING_LINK = "missing_link"
    CONTRADICTION = "contradiction"


@dataclass(frozen=True)
class SemanticFinding:
    finding_type: SemanticFindingType
    explanation: str
    supporting_chunk_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.finding_type, SemanticFindingType):
            raise TypeError("Invalid semantic finding type.")

        if not isinstance(self.explanation, str) or not self.explanation.strip():
            raise ValueError("Finding explanation must be nonempty.")

        if not isinstance(self.supporting_chunk_ids, tuple):
            raise TypeError("Supporting chunk IDs must be a tuple.")

        if len(set(self.supporting_chunk_ids)) != len(self.supporting_chunk_ids):
            raise ValueError("Duplicate supporting chunk IDs.")

        if any(
            not isinstance(chunk_id, str) or not chunk_id.strip()
            for chunk_id in self.supporting_chunk_ids
        ):
            raise ValueError("Supporting chunk IDs must be nonempty strings.")

        if (
            self.finding_type
            in (SemanticFindingType.SUPPORT, SemanticFindingType.CONTRADICTION)
            and not self.supporting_chunk_ids
        ):
            raise ValueError(
                "Support and contradiction findings require cited chunks."
            )


@dataclass(frozen=True)
class SemanticEvidenceAssessment:
    support: SemanticSupport
    findings: tuple[SemanticFinding, ...]
    assessed_chunk_ids: tuple[str, ...]
    explanation: str
    assessor_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.support, SemanticSupport):
            raise TypeError("Invalid semantic support state.")

        if not isinstance(self.findings, tuple):
            raise TypeError("Findings must be a tuple.")

        if any(not isinstance(item, SemanticFinding) for item in self.findings):
            raise TypeError("Every finding must be a SemanticFinding.")

        if not isinstance(self.assessed_chunk_ids, tuple):
            raise TypeError("Assessed chunk IDs must be a tuple.")

        if len(set(self.assessed_chunk_ids)) != len(self.assessed_chunk_ids):
            raise ValueError("Duplicate assessed chunk IDs.")

        if any(
            not isinstance(chunk_id, str) or not chunk_id.strip()
            for chunk_id in self.assessed_chunk_ids
        ):
            raise ValueError("Assessed chunk IDs must be nonempty strings.")

        if not isinstance(self.explanation, str) or not self.explanation.strip():
            raise ValueError("Assessment explanation must be nonempty.")

        if not isinstance(self.assessor_name, str) or not self.assessor_name.strip():
            raise ValueError("Assessor name must be nonempty.")

        allowed = set(self.assessed_chunk_ids)

        for finding in self.findings:
            if not set(finding.supporting_chunk_ids).issubset(allowed):
                raise ValueError(
                    "Finding references chunks outside the assessed evidence."
                )

        if self.support == SemanticSupport.SUPPORTED and not any(
            finding.finding_type == SemanticFindingType.SUPPORT
            for finding in self.findings
        ):
            raise ValueError(
                "SUPPORTED assessment requires a cited support finding."
            )

    @property
    def has_contradiction(self) -> bool:
        return any(
            finding.finding_type == SemanticFindingType.CONTRADICTION
            for finding in self.findings
        )

    @property
    def has_missing_link(self) -> bool:
        return any(
            finding.finding_type == SemanticFindingType.MISSING_LINK
            for finding in self.findings
        )
