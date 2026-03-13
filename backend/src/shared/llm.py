"""LiteLLM wrapper for model-agnostic LLM calls."""

import logging

import litellm

logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True


class LLMClient:
    def __init__(self, model: str = "gpt-4o-mini", timeout: int = 30):
        self.model = model
        self.timeout = timeout

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=temperature,
            timeout=self.timeout,
        )

        result = response.choices[0].message.content
        tokens = response.usage.total_tokens
        logger.debug("LLM call: model=%s, tokens=%d", self.model, tokens)

        return result
