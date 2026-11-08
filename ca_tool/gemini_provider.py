"""Gemini Developer API transport for Google AI Studio API keys."""

from __future__ import annotations

import asyncio
import math
import re

from .provider import ProviderError, ProviderReply
from .settings import Settings
from .vertex_provider import VertexAIProvider

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider(VertexAIProvider):
    """Reuse Google's native JSON contract with independent key-only routing."""

    def __init__(self, settings: Settings):
        # Do not initialize Vertex routing or inspect its account/auth settings.
        if settings.provider != "gemini":
            raise ProviderError("CONFIG_ERROR", "Unsupported LLM_PROVIDER for Gemini Developer API")
        if (not isinstance(settings.model, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", settings.model)):
            raise ProviderError("CONFIG_ERROR", "LLM_MODEL must be a bare Google model ID")
        if not isinstance(settings.api_key, str) or not settings.api_key.strip():
            raise ProviderError("CONFIG_ERROR", "Set the configured Gemini Developer API key")
        if settings.base_url != GEMINI_BASE_URL:
            raise ProviderError("CONFIG_ERROR", "Gemini LLM_BASE_URL must use the canonical Google Developer API")
        self.settings = settings
        self._endpoint = f"{settings.base_url}/models/{settings.model}:generateContent"

    def complete(self, messages: list[dict], schema: dict, timeout: float) -> ProviderReply:
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ProviderError("CONFIG_ERROR", "Provider timeout must be finite and positive")
        body = self._request_body(messages, schema)
        headers = {"x-goog-api-key": self.settings.api_key}
        return asyncio.run(self._complete(body, headers, timeout))
