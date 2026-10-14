
"""Phase 11.4E-3: independent chain-level claim verification.

Checks each proposed reasoning relationship against its cited text.
Diagnostic only: does not establish chain completeness or authorize
automatic recovery acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.recovery.conservative_entailment_verifier import (
    ConservativeEntailmentVerifier,
)
from src.recovery.evidence_entailment import (
    ChainEntailmentAssessment,
    ClaimVerification,
    EntailmentLabel,
)
from src.recovery.reasoning_chain import (
    ReasoningChainAssessment,
    ReasoningStepStatus,
)
from src.recovery.semantic_verification import SemanticSupport


@dataclass(frozen=True)
class ChainVerificationResult:
    assessment: ChainEntailmentAssessment
    proposed_support: SemanticSupport
    passes_literal_gate: bool
    explanation: str


class ChainEntailmentVerifier:
    name = "chain_literal_verification_v1"

    def __init__(self) -> None:
        self.claim_verifier = ConservativeEntailmentVerifier()

    def verify(
        self,
        chain: ReasoningChainAssessment,
        evidence_by_id: Mapping[str, str],
    ) -> ChainVerificationResult:
        if not isinstance(chain, ReasoningChainAssessment):
            raise TypeError("Expected ReasoningChainAssessment.")

        if not isinstance(evidence_by_id, Mapping):
            raise TypeError("evidence_by_id must be a mapping.")

        # Require exactly the evidence IDs recorded by the assessor.
        # This prevents silent verification against extra evidence.
        if set(evidence_by_id) != set(chain.assessed_chunk_ids):
            raise ValueError(
                "Evidence IDs must match assessed_chunk_ids."
            )

        if any(
            not isinstance(text, str)
            for text in evidence_by_id.values()
        ):
            raise TypeError("Evidence values must be strings.")

        verifications: list[ClaimVerification] = []

        for step in chain.steps:
            if step.status != ReasoningStepStatus.SUPPORTED:
                verifications.append(
                    ClaimVerification(
                        claim=step.relationship,
                        label=EntailmentLabel.UNKNOWN,
                        cited_chunk_ids=step.supporting_chunk_ids,
                        explanation=(
                            "Step was not proposed as supported; "
                            "literal entailment was not attempted."
                        ),
                        verifier_name=self.name,
                    )
                )
                continue

            result = self.claim_verifier.verify(
                claim=step.relationship,
                cited_chunk_ids=step.supporting_chunk_ids,
                evidence_by_id=evidence_by_id,
            )
            verifications.append(result)

        assessment = ChainEntailmentAssessment(
            claim_verifications=tuple(verifications),
            assessed_chunk_ids=chain.assessed_chunk_ids,
            explanation=(
                "Independent literal verification of each "
                "proposed reasoning step."
            ),
        )

        passes = (
            chain.support == SemanticSupport.SUPPORTED
            and bool(chain.candidate_answer)
            and bool(chain.steps)
            and not chain.contradictions
            and assessment.all_entailed
        )

        if passes:
            explanation = (
                "All proposed steps matched cited evidence "
                "literally. Chain completeness, omitted conflicts, "
                "and answer correctness remain unverified."
            )
        else:
            explanation = (
                "The proposed chain did not pass the "
                "conservative literal verification gate."
            )

        return ChainVerificationResult(
            assessment=assessment,
            proposed_support=chain.support,
            passes_literal_gate=passes,
            explanation=explanation,
        )
