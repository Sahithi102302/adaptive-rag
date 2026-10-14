
"""Phase 11.4E-1: evidence entailment verification contracts.

These contracts represent the outcome of checking a factual claim
against cited evidence. They do not perform semantic inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EntailmentLabel(str, Enum):
    ENTAILED = "entailed"
    CONTRADICTED = "contradicted"
    NOT_ESTABLISHED = "not_established"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClaimVerification:
    claim: str
    label: EntailmentLabel
    cited_chunk_ids: tuple[str, ...]
    explanation: str
    verifier_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.claim, str) or not self.claim.strip():
            raise ValueError("Claim must be nonempty.")

        if not isinstance(self.label, EntailmentLabel):
            raise TypeError("label must be an EntailmentLabel.")

        if not isinstance(self.cited_chunk_ids, tuple):
            raise TypeError("cited_chunk_ids must be a tuple.")

        if any(
            not isinstance(chunk_id, str) or not chunk_id.strip()
            for chunk_id in self.cited_chunk_ids
        ):
            raise ValueError("Invalid cited chunk ID.")

        if len(set(self.cited_chunk_ids)) != len(self.cited_chunk_ids):
            raise ValueError("Duplicate cited chunk IDs.")

        if (
            self.label != EntailmentLabel.UNKNOWN
            and not self.cited_chunk_ids
        ):
            raise ValueError(
                "A definite verification label requires cited evidence."
            )

        if (
            not isinstance(self.explanation, str)
            or not self.explanation.strip()
        ):
            raise ValueError("Explanation must be nonempty.")

        if (
            not isinstance(self.verifier_name, str)
            or not self.verifier_name.strip()
        ):
            raise ValueError("Verifier name must be nonempty.")


@dataclass(frozen=True)
class ChainEntailmentAssessment:
    claim_verifications: tuple[ClaimVerification, ...]
    assessed_chunk_ids: tuple[str, ...]
    explanation: str

    def __post_init__(self) -> None:
        if not isinstance(self.claim_verifications, tuple):
            raise TypeError("claim_verifications must be a tuple.")

        if not all(
            isinstance(item, ClaimVerification)
            for item in self.claim_verifications
        ):
            raise TypeError("Invalid claim verification object.")

        if not isinstance(self.assessed_chunk_ids, tuple):
            raise TypeError("assessed_chunk_ids must be a tuple.")

        if any(
            not isinstance(chunk_id, str) or not chunk_id.strip()
            for chunk_id in self.assessed_chunk_ids
        ):
            raise ValueError("Invalid assessed chunk ID.")

        if len(set(self.assessed_chunk_ids)) != len(
            self.assessed_chunk_ids
        ):
            raise ValueError("Duplicate assessed chunk IDs.")

        allowed_ids = set(self.assessed_chunk_ids)

        for verification in self.claim_verifications:
            if not set(
                verification.cited_chunk_ids
            ).issubset(allowed_ids):
                raise ValueError(
                    "Claim verification cites unknown evidence."
                )

        if (
            not isinstance(self.explanation, str)
            or not self.explanation.strip()
        ):
            raise ValueError("Explanation must be nonempty.")

    @property
    def all_entailed(self) -> bool:
        return bool(self.claim_verifications) and all(
            item.label == EntailmentLabel.ENTAILED
            for item in self.claim_verifications
        )

    @property
    def has_contradiction(self) -> bool:
        return any(
            item.label == EntailmentLabel.CONTRADICTED
            for item in self.claim_verifications
        )

    @property
    def has_unverified_claims(self) -> bool:
        return any(
            item.label in (
                EntailmentLabel.NOT_ESTABLISHED,
                EntailmentLabel.UNKNOWN,
            )
            for item in self.claim_verifications
        )
