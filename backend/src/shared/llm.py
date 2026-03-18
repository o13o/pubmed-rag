"""LiteLLM wrapper for model-agnostic LLM calls."""

import logging
import os
from collections.abc import Generator

import litellm

logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True

_langfuse_initialized = False


def _init_langfuse() -> None:
    """Enable LangFuse tracing if credentials are configured."""
    global _langfuse_initialized
    if _langfuse_initialized:
        return
    _langfuse_initialized = True

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    if not public_key:
        # LiteLLM's langfuse callback reads credentials from os.environ,
        # so we copy them from pydantic-settings (.env) if not already set.
        from src.shared.config import get_settings
        settings = get_settings()
        if settings.langfuse_public_key:
            os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
            os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
            os.environ["LANGFUSE_HOST"] = settings.langfuse_host
            public_key = settings.langfuse_public_key

    if public_key:
        if "langfuse" not in litellm.success_callback:
            litellm.success_callback.append("langfuse")
        if "langfuse" not in litellm.failure_callback:
            litellm.failure_callback.append("langfuse")
        logger.info("LangFuse tracing enabled")


class LLMClient:
    def __init__(self, model: str = "gpt-4o-mini", timeout: int = 30, num_retries: int = 3):
        _init_langfuse()
        self.model = model
        self.timeout = timeout
        self.num_retries = num_retries

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
            num_retries=self.num_retries,
        )

        result = response.choices[0].message.content
        tokens = response.usage.total_tokens
        logger.debug("LLM call: model=%s, tokens=%d", self.model, tokens)

        return result

    def complete_stream(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.0,
    ) -> Generator[str, None, None]:
        """Yield text chunks from LLM streaming response."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=temperature,
            timeout=self.timeout,
            num_retries=self.num_retries,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in response:
            delta = (
                getattr(chunk.choices[0].delta, "content", None)
                if chunk.choices
                else None
            )
            if delta:
                yield delta
            if hasattr(chunk, "usage") and chunk.usage:
                logger.debug(
                    "LLM stream: model=%s, tokens=%d",
                    self.model,
                    chunk.usage.total_tokens,
                )
