"""LiteLLM wrapper for model-agnostic LLM calls."""

import logging
import os

import litellm

logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True

# Enable LangFuse tracing if credentials are configured
if os.environ.get("LANGFUSE_PUBLIC_KEY"):
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]
    logger.info("LangFuse tracing enabled")


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
