
"""Phase 11.4G-8: Qwen-assisted question decomposition.

Generates structured relationship proposals using
Qwen2.5-1.5B-Instruct.

Validation pipeline:
    Qwen generation
        -> JSON parsing
        -> structural validation
        -> task-specific semantic consistency gate
        -> validated decomposition proposal

Important limitations:
- The semantic gate supports only a narrow question family.
- Passing the gate does not establish general semantic fidelity.
- Retrieved evidence and gold answers are not used.
- Automatic answer acceptance remains disabled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.recovery.decomposition_proposal_validator import (
    DecompositionProposalValidator,
    ProposalValidation,
)

from src.recovery.decomposition_semantic_gate import (
    DecompositionSemanticGate,
)


class QwenDecompositionStatus(str, Enum):
    VALID_PROPOSAL = "valid_proposal"
    ABSTAINED = "abstained"
    INVALID_JSON = "invalid_json"
    INVALID_PROPOSAL = "invalid_proposal"
    SEMANTICALLY_REJECTED = "semantically_rejected"


@dataclass(frozen=True)
class QwenDecompositionResult:
    status: QwenDecompositionStatus
    raw_response: str
    proposal: dict[str, Any] | None
    validation: ProposalValidation | None
    explanation: str

    @property
    def is_structurally_valid(self) -> bool:
        """Whether the proposal passed structural validation."""
        return (
            self.validation is not None
            and self.validation.is_valid
        )

    @property
    def is_usable(self) -> bool:
        """Whether the proposal passed both validation stages."""
        return (
            self.status == QwenDecompositionStatus.VALID_PROPOSAL
            and self.is_structurally_valid
        )


class QwenQuestionDecomposer:
    name = "qwen_question_decomposer_v2"

    SYSTEM_PROMPT = """You decompose questions into factual relationships.

Supported predicates:
- authored_by: book -> author
- attended: person -> university or college

Return exactly one JSON object.

For a supported question, use:
{
  "relationships": [
    {
      "subject": "Book title",
      "predicate": "authored_by",
      "object": "?author"
    },
    {
      "subject": "?author",
      "predicate": "attended",
      "object": "?university"
    }
  ],
  "answer_variable": "?university"
}

For questions that cannot be faithfully represented using
the supported predicates, return:
{"abstain": true}

Do not invent entities.
Do not use retrieved evidence.
Do not answer the question.
Do not add markdown fences or explanations.
"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens: int = 256,
        local_files_only: bool = True,
    ) -> None:
        import torch

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        self.torch = torch
        self.max_new_tokens = max_new_tokens

        # Stage 1: Structural validation.
        self.validator = DecompositionProposalValidator()

        # Stage 2: Task-specific semantic consistency gate.
        self.semantic_gate = DecompositionSemanticGate()

        # Load tokenizer from local Hugging Face cache.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )

        # Load Qwen model.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            dtype=torch.float32,
        )

        self.model.eval()

    @staticmethod
    def _parse_json(raw: str) -> Any:
        """Parse a complete JSON response.

        Optional Markdown code fences are tolerated.
        Extra text outside the JSON object is rejected.
        """
        text = raw.strip()

        if text.startswith("```"):
            lines = text.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            text = "\n".join(lines).strip()

        return json.loads(text)

    def decompose(
        self,
        question: str,
    ) -> QwenDecompositionResult:
        """Generate and validate a question decomposition."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be nonempty.")

        messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        # Tokenize the chat conversation.
        # return_dict=True avoids the previous BatchEncoding
        # versus tensor compatibility issue.
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        # Ensure tensors are on the model's device.
        device = next(self.model.parameters()).device

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        input_length = encoded["input_ids"].shape[-1]

        # Deterministic generation.
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only newly generated tokens.
        generated = outputs[0][input_length:]

        raw = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        # --------------------------------------------
        # Stage 1: Parse JSON
        # --------------------------------------------
        try:
            proposal = self._parse_json(raw)

        except (json.JSONDecodeError, ValueError) as exc:
            return QwenDecompositionResult(
                status=QwenDecompositionStatus.INVALID_JSON,
                raw_response=raw,
                proposal=None,
                validation=None,
                explanation=f"JSON parsing failed: {exc}",
            )

        # --------------------------------------------
        # Stage 2: Explicit model abstention
        # --------------------------------------------
        if (
            isinstance(proposal, dict)
            and proposal == {"abstain": True}
        ):
            return QwenDecompositionResult(
                status=QwenDecompositionStatus.ABSTAINED,
                raw_response=raw,
                proposal=proposal,
                validation=None,
                explanation="Model explicitly abstained.",
            )

        # --------------------------------------------
        # Stage 3: Structural proposal validation
        # --------------------------------------------
        validation = self.validator.validate(
            question,
            proposal,
        )

        if not validation.is_valid:
            return QwenDecompositionResult(
                status=QwenDecompositionStatus.INVALID_PROPOSAL,
                raw_response=raw,
                proposal=(
                    proposal
                    if isinstance(proposal, dict)
                    else None
                ),
                validation=validation,
                explanation=validation.explanation,
            )

        # --------------------------------------------
        # Stage 4: Semantic consistency gate
        # --------------------------------------------
        # A structurally valid graph may still misrepresent
        # the meaning of the original question.
        #
        # This gate checks consistency with our supported
        # two-hop author/university question grammar.
        semantic_result = self.semantic_gate.check(
            question,
            validation.requirements,
        )

        if not semantic_result.accepted:
            return QwenDecompositionResult(
                status=(
                    QwenDecompositionStatus.SEMANTICALLY_REJECTED
                ),
                raw_response=raw,
                proposal=proposal,
                validation=validation,
                explanation=(
                    "Semantic consistency gate rejected proposal: "
                    f"{semantic_result.reason}"
                ),
            )

        # --------------------------------------------
        # Stage 5: Validated decomposition proposal
        # --------------------------------------------
        return QwenDecompositionResult(
            status=QwenDecompositionStatus.VALID_PROPOSAL,
            raw_response=raw,
            proposal=proposal,
            validation=validation,
            explanation=(
                "Proposal passed structural validation and "
                "the task-specific semantic consistency gate. "
                "General semantic fidelity remains unverified."
            ),
        )
