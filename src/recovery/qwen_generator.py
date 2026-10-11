from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


@dataclass(frozen=True)
class QwenGeneratorConfig:
    """
    Configuration for the local Qwen text-generation
    backend used in bridge-query experiments.
    """

    model_name: str = (
        "Qwen/Qwen2.5-1.5B-Instruct"
    )

    max_new_tokens: int = 128

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be positive."
            )


class QwenTextGenerator:
    """
    Local CPU text-generation adapter for the
    relation-aware bridge-query experiment.

    This component knows nothing about:
    - HotpotQA gold evidence
    - gold answers
    - oracle failure labels
    - retrieval evaluation
    - recovery success

    It implements only:

        prompt -> generated text

    This keeps the language-model backend separate
    from the bridge-query and recovery logic.
    """

    def __init__(
        self,
        config: QwenGeneratorConfig | None = None,
    ) -> None:
        self.config = (
            config
            or QwenGeneratorConfig()
        )

        print(
            "Loading Qwen tokenizer..."
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.config.model_name
            )
        )

        print(
            "Loading Qwen model on CPU..."
        )

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                dtype=torch.float32,
            )
        )

        self.model.to(
            "cpu"
        )

        self.model.eval()

    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Generate deterministic text from a prompt.

        The prompt is wrapped using the model's
        instruction/chat template.

        Only newly generated tokens are decoded;
        prompt tokens are excluded from the returned
        string.
        """

        if not isinstance(
            prompt,
            str,
        ):
            raise TypeError(
                "prompt must be a string."
            )

        prompt = prompt.strip()

        if not prompt:
            raise ValueError(
                "prompt cannot be empty."
            )

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        formatted_prompt = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        model_inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
        )

        model_inputs = {
            key: value.to("cpu")
            for key, value
            in model_inputs.items()
        }

        generation_kwargs = {
            "max_new_tokens": (
                self.config.max_new_tokens
            ),
            "do_sample": False,
        }

        with torch.inference_mode():
            generated_ids = (
                self.model.generate(
                    **model_inputs,
                    **generation_kwargs,
                )
            )

        input_length = (
            model_inputs[
                "input_ids"
            ].shape[1]
        )

        generated_length = (
            generated_ids.shape[1]
        )

        new_token_count = (
            generated_length
            - input_length
        )

        # Temporary experiment-time diagnostics.
        # Keep these during Phase 10.3D-Real so we
        # can inspect generation behavior.
        print(
            "\n[Qwen generation diagnostics]"
        )

        print(
            f"input_tokens={input_length}"
        )

        print(
            f"total_tokens={generated_length}"
        )

        print(
            f"new_tokens={new_token_count}"
        )

        new_tokens = generated_ids[
            0,
            input_length:,
        ]

        output = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        )

        return output.strip()