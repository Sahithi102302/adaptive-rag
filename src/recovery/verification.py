
"""Post-recovery verification contracts for AdaptiveRAG.

Phase 11.1 defines verification states and structural invariants.
It does not decide whether recovered evidence is semantically sufficient.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple


class VerificationDecision(str, Enum):
    """Possible post-recovery verification decisions."""

    ACCEPT = "accept"
    REJECT = "reject"
    UNCERTAIN = "uncertain"


class VerificationStatus(str, Enum):
    """Whether an actual verification assessment has been performed."""

    NOT_EVALUATED = "not_evaluated"
    EVALUATED = "evaluated"


@dataclass(frozen=True)
class VerificationResult:
    """Immutable record of a post-recovery verification assessment."""

    status: VerificationStatus
    decision: VerificationDecision
    reason: str
    original_results: Tuple[Any, ...]
    candidate_results: Tuple[Any, ...]
    metadata: Optional[dict] = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, VerificationStatus):
            raise TypeError("status must be a VerificationStatus")

        if not isinstance(self.decision, VerificationDecision):
            raise TypeError("decision must be a VerificationDecision")

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("Verification reason must be nonempty")

        if not isinstance(self.original_results, tuple):
            raise TypeError("original_results must be a tuple")

        if not isinstance(self.candidate_results, tuple):
            raise TypeError("candidate_results must be a tuple")

        if self.status == VerificationStatus.NOT_EVALUATED:
            if self.decision != VerificationDecision.UNCERTAIN:
                raise ValueError(
                    "Unevaluated evidence cannot be accepted or rejected"
                )

        if self.decision == VerificationDecision.ACCEPT:
            if self.status != VerificationStatus.EVALUATED:
                raise ValueError(
                    "Acceptance requires completed verification"
                )

            if not self.candidate_results:
                raise ValueError(
                    "Cannot accept an empty recovery candidate"
                )

    @property
    def accepted(self) -> bool:
        return (
            self.status == VerificationStatus.EVALUATED
            and self.decision == VerificationDecision.ACCEPT
        )

    @property
    def requires_fallback(self) -> bool:
        return (
            self.status == VerificationStatus.EVALUATED
            and self.decision == VerificationDecision.REJECT
        )
