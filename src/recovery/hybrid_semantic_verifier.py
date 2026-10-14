
"""Phase 11.4G-12A: conservative hybrid semantic router.

This module separates:
- deterministic semantic acceptance,
- structurally invalid proposals,
- semantic-gate rejections,
- unresolved out-of-grammar proposals.

It does not implement a learned semantic verifier yet.
It never automatically accepts an out-of-grammar proposal.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.recovery.decomposition_proposal_validator import (
    DecompositionProposalValidator,
)
from src.recovery.decomposition_semantic_gate import (
    DecompositionSemanticGate,
)


class HybridDecision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    ABSTAIN = "abstain"


class HybridReason(str, Enum):
    VERIFIED_GRAMMAR = "verified_grammar"
    INVALID_STRUCTURE = "invalid_structure"
    IN_GRAMMAR_MISMATCH = "in_grammar_mismatch"
    OUT_OF_GRAMMAR = "out_of_grammar"
    INVALID_INPUT = "invalid_input"


@dataclass(frozen=True)
class HybridVerificationResult:
    decision: HybridDecision
    reason: HybridReason
    explanation: str

    @property
    def accepted(self) -> bool:
        return self.decision == HybridDecision.ACCEPT


class HybridSemanticVerifier:
    name = "hybrid_semantic_verifier_v1"

    def __init__(self) -> None:
        self.validator = DecompositionProposalValidator()
        self.semantic_gate = DecompositionSemanticGate()

    def verify(
        self,
        question: str,
        proposal: Any,
    ) -> HybridVerificationResult:
        if not isinstance(question, str) or not question.strip():
            return HybridVerificationResult(
                decision=HybridDecision.REJECT,
                reason=HybridReason.INVALID_INPUT,
                explanation="Question must be nonempty.",
            )

        validation = self.validator.validate(
            question,
            proposal,
        )

        if not validation.is_valid:
            return HybridVerificationResult(
                decision=HybridDecision.REJECT,
                reason=HybridReason.INVALID_STRUCTURE,
                explanation=validation.explanation,
            )

        gate_result = self.semantic_gate.check(
            question,
            validation.requirements,
        )

        if gate_result.accepted:
            return HybridVerificationResult(
                decision=HybridDecision.ACCEPT,
                reason=HybridReason.VERIFIED_GRAMMAR,
                explanation=gate_result.reason,
            )

        if (
            "outside the supported two-hop" in
            gate_result.reason.casefold()
        ):
            return HybridVerificationResult(
                decision=HybridDecision.ABSTAIN,
                reason=HybridReason.OUT_OF_GRAMMAR,
                explanation=(
                    "Structurally valid proposal, but the "
                    "question is outside the verified grammar. "
                    "No fallback acceptance is authorized."
                ),
            )

        return HybridVerificationResult(
            decision=HybridDecision.REJECT,
            reason=HybridReason.IN_GRAMMAR_MISMATCH,
            explanation=gate_result.reason,
        )
