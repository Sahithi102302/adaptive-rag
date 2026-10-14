
"""Phase 11.4E-2: conservative literal-evidence verifier.

A deliberately narrow baseline:
- Matches complete claims to explicit evidence sentences.
- Ignores capitalization and minor formatting differences.
- Does not infer relationships from token overlap.
- Does not perform general semantic entailment or contradiction detection.

No model calls or automatic recovery acceptance.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Mapping, Sequence

from src.recovery.evidence_entailment import (
    ClaimVerification,
    EntailmentLabel,
)


class ConservativeEntailmentVerifier:
    name = "conservative_literal_verifier_v1"

    @staticmethod
    def _normalize(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = text.casefold()
        text = re.sub(r"\s+", " ", text).strip()
        return text.rstrip(".!?").strip()

    @classmethod
    def _sentences(cls, text: str) -> tuple[str, ...]:
        # This is a simple English sentence splitter.
        # Abbreviations and complex punctuation are not supported.
        parts = re.split(r"(?<=[.!?])\s+|\n+", text)
        return tuple(
            normalized
            for part in parts
            if (normalized := cls._normalize(part))
        )

    def verify(
        self,
        claim: str,
        cited_chunk_ids: Sequence[str],
        evidence_by_id: Mapping[str, str],
    ) -> ClaimVerification:
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("Claim must be nonempty.")

        if not isinstance(evidence_by_id, Mapping):
            raise TypeError("evidence_by_id must be a mapping.")

        citations = tuple(cited_chunk_ids)

        if any(
            not isinstance(chunk_id, str) or not chunk_id.strip()
            for chunk_id in citations
        ):
            raise ValueError("Invalid citation ID.")

        if len(set(citations)) != len(citations):
            raise ValueError("Duplicate citation IDs.")

        for chunk_id in citations:
            if chunk_id not in evidence_by_id:
                raise ValueError(
                    f"Unknown cited chunk ID: {chunk_id}"
                )

            if not isinstance(evidence_by_id[chunk_id], str):
                raise TypeError("Evidence text must be a string.")

        if not citations:
            return ClaimVerification(
                claim=claim,
                label=EntailmentLabel.UNKNOWN,
                cited_chunk_ids=(),
                explanation="No cited evidence was supplied.",
                verifier_name=self.name,
            )

        normalized_claim = self._normalize(claim)

        for chunk_id in citations:
            sentences = self._sentences(
                evidence_by_id[chunk_id]
            )

            if normalized_claim in sentences:
                return ClaimVerification(
                    claim=claim,
                    label=EntailmentLabel.ENTAILED,
                    cited_chunk_ids=citations,
                    explanation=(
                        "The complete normalized claim matches "
                        f"an explicit sentence in chunk {chunk_id}."
                    ),
                    verifier_name=self.name,
                )

        return ClaimVerification(
            claim=claim,
            label=EntailmentLabel.NOT_ESTABLISHED,
            cited_chunk_ids=citations,
            explanation=(
                "No cited passage contains an exact normalized "
                "sentence matching the claim. This does not "
                "prove the claim is false."
            ),
            verifier_name=self.name,
        )
