
"""Phase 11.4D-1: reasoning-chain verification contracts.

Defines structural invariants for evidence-grounded reasoning chains.

No LLM inference, gold annotations, recovery decisions,
or automatic acceptance are performed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.recovery.semantic_verification import SemanticSupport


class ReasoningStepStatus(str, Enum):
    SUPPORTED = "supported"
    MISSING = "missing"
    CONFLICTING = "conflicting"
    UNKNOWN = "unknown"


def _validate_chunk_ids(
    chunk_ids: tuple[str, ...],
    field_name: str,
) -> None:
    if not isinstance(chunk_ids, tuple):
        raise TypeError(f"{field_name} must be a tuple.")

    if any(
        not isinstance(chunk_id, str) or not chunk_id.strip()
        for chunk_id in chunk_ids
    ):
        raise ValueError(
            f"{field_name} contains an invalid chunk ID."
        )

    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError(
            f"{field_name} contains duplicate chunk IDs."
        )


@dataclass(frozen=True)
class ReasoningStep:
    step_id: int
    relationship: str
    status: ReasoningStepStatus
    supporting_chunk_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.step_id) is not int or self.step_id <= 0:
            raise ValueError(
                "step_id must be a positive integer."
            )

        if (
            not isinstance(self.relationship, str)
            or not self.relationship.strip()
        ):
            raise ValueError(
                "A reasoning step requires a relationship."
            )

        if not isinstance(self.status, ReasoningStepStatus):
            raise TypeError(
                "status must be a ReasoningStepStatus."
            )

        _validate_chunk_ids(
            self.supporting_chunk_ids,
            "supporting_chunk_ids",
        )

        if (
            self.status == ReasoningStepStatus.SUPPORTED
            and not self.supporting_chunk_ids
        ):
            raise ValueError(
                "A supported step requires evidence citations."
            )

        if (
            self.status in (
                ReasoningStepStatus.MISSING,
                ReasoningStepStatus.UNKNOWN,
            )
            and self.supporting_chunk_ids
        ):
            raise ValueError(
                "Missing or unknown steps cannot claim "
                "supporting citations."
            )

        if (
            self.status == ReasoningStepStatus.CONFLICTING
            and len(self.supporting_chunk_ids) < 2
        ):
            raise ValueError(
                "A conflicting step requires at least "
                "two distinct cited chunks."
            )


@dataclass(frozen=True)
class EvidenceContradiction:
    explanation: str
    supporting_chunk_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.explanation, str)
            or not self.explanation.strip()
        ):
            raise ValueError(
                "A contradiction requires an explanation."
            )

        _validate_chunk_ids(
            self.supporting_chunk_ids,
            "supporting_chunk_ids",
        )

        if len(self.supporting_chunk_ids) < 2:
            raise ValueError(
                "A contradiction requires at least "
                "two distinct cited chunks."
            )


@dataclass(frozen=True)
class ReasoningChainAssessment:
    candidate_answer: str | None
    steps: tuple[ReasoningStep, ...]
    contradictions: tuple[EvidenceContradiction, ...]
    support: SemanticSupport
    assessed_chunk_ids: tuple[str, ...]
    explanation: str
    assessor_name: str

    def __post_init__(self) -> None:
        if (
            self.candidate_answer is not None
            and (
                not isinstance(self.candidate_answer, str)
                or not self.candidate_answer.strip()
            )
        ):
            raise ValueError(
                "candidate_answer must be nonempty or None."
            )

        if not isinstance(self.steps, tuple):
            raise TypeError("steps must be a tuple.")

        if not all(
            isinstance(step, ReasoningStep)
            for step in self.steps
        ):
            raise TypeError(
                "All steps must be ReasoningStep objects."
            )

        if not isinstance(self.contradictions, tuple):
            raise TypeError(
                "contradictions must be a tuple."
            )

        if not all(
            isinstance(item, EvidenceContradiction)
            for item in self.contradictions
        ):
            raise TypeError(
                "Invalid contradiction object."
            )

        if not isinstance(self.support, SemanticSupport):
            raise TypeError(
                "support must be a SemanticSupport value."
            )

        _validate_chunk_ids(
            self.assessed_chunk_ids,
            "assessed_chunk_ids",
        )

        if (
            not isinstance(self.explanation, str)
            or not self.explanation.strip()
        ):
            raise ValueError(
                "An assessment requires an explanation."
            )

        if (
            not isinstance(self.assessor_name, str)
            or not self.assessor_name.strip()
        ):
            raise ValueError(
                "An assessment requires an assessor name."
            )

        step_ids = [step.step_id for step in self.steps]

        if len(set(step_ids)) != len(step_ids):
            raise ValueError(
                "Reasoning step IDs must be unique."
            )

        allowed_ids = set(self.assessed_chunk_ids)

        for step in self.steps:
            if not set(
                step.supporting_chunk_ids
            ).issubset(allowed_ids):
                raise ValueError(
                    "Reasoning step cites unknown evidence."
                )

        for contradiction in self.contradictions:
            if not set(
                contradiction.supporting_chunk_ids
            ).issubset(allowed_ids):
                raise ValueError(
                    "Contradiction cites unknown evidence."
                )

        has_conflicting_step = any(
            step.status == ReasoningStepStatus.CONFLICTING
            for step in self.steps
        )

        if (
            has_conflicting_step
            and not self.contradictions
        ):
            raise ValueError(
                "Conflicting steps require an explicit "
                "contradiction record."
            )

        if self.support == SemanticSupport.SUPPORTED:
            if self.candidate_answer is None:
                raise ValueError(
                    "SUPPORTED requires a candidate answer."
                )

            if not self.steps:
                raise ValueError(
                    "SUPPORTED requires reasoning steps."
                )

            if any(
                step.status != ReasoningStepStatus.SUPPORTED
                for step in self.steps
            ):
                raise ValueError(
                    "SUPPORTED requires every step "
                    "to be supported."
                )

            if self.contradictions:
                raise ValueError(
                    "SUPPORTED cannot contain contradictions."
                )

    @property
    def has_missing_steps(self) -> bool:
        return any(
            step.status == ReasoningStepStatus.MISSING
            for step in self.steps
        )

    @property
    def has_conflicts(self) -> bool:
        return bool(self.contradictions) or any(
            step.status == ReasoningStepStatus.CONFLICTING
            for step in self.steps
        )
