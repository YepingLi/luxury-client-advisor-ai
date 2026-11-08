"""One configured Chat Completions call via the official OpenAI SDK transport.

The application owns all retries. There is no fallback model, canned draft, or
automatic correction hidden in this adapter.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from openai import AsyncOpenAI, APIConnectionError, APIStatusError, APITimeoutError

from .settings import Settings


@dataclass
class ProviderReply:
    content: str | dict | None
    raw: dict[str, Any]
    response_id: str | None = None
    usage: dict[str, Any] | None = None
    model: str | None = None
    finish_reason: str | None = "stop"
    refusal: str | None = None


class ProviderError(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = False,
                 retry_after: float | None = None, status_code: int | None = None,
                 response_id: str | None = None, error_category: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.retry_after = retry_after
        self.status_code = status_code
        self.response_id = response_id
        self.error_category = error_category

    def as_dict(self):
        return {"code": self.code, "message": self.message, "retryable": self.retryable,
                "retry_after": self.retry_after, "status_code": self.status_code,
                "response_id": self.response_id, "error_category": self.error_category}


def classify_gateway_error(response) -> str | None:
    """Return a fixed diagnostic label; never copy provider text into logs."""
    body = response.text.lower()
    if any(term in body for term in ("cloudflare", "web application firewall")):
        return "gateway_firewall"
    if any(term in body for term in ("invalid api key", "invalid_api_key", "incorrect api key", "invalid token")):
        return "invalid_credentials"
    if "insufficient" in body and any(term in body for term in ("credit", "balance", "quota")):
        return "insufficient_credits"
    if "model" in body and any(term in body for term in ("not allowed", "permission", "not authorized", "not permitted")):
        return "model_access_denied"
    return None


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value)
        if seconds >= 0 and seconds < float("inf"):
            return seconds
    except ValueError:
        pass
    try:
        date = parsedate_to_datetime(value)
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
    except (ValueError, TypeError, OverflowError):
        return None


def create_provider(settings: Settings):
    if settings.provider == "gemini":
        from .gemini_provider import GeminiProvider
        return GeminiProvider(settings)
    if settings.provider == "vertex":
        from .vertex_provider import VertexAIProvider
        return VertexAIProvider(settings)
    return OpenRouterProvider(settings)


class OpenRouterProvider:
    source = "live"

    def __init__(self, settings: Settings):
        if settings.provider not in {"openrouter", "tokenrouter", "openai_compatible"}:
            raise ProviderError("CONFIG_ERROR", "Unsupported LLM_PROVIDER")
        if not settings.api_key.strip() or not settings.model.strip():
            raise ProviderError("CONFIG_ERROR", "Set the configured provider API key and LLM_MODEL in .env")
        self.settings = settings

    def request_parameters(self, schema: dict) -> dict:
        params = {
            "model": self.settings.model,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "wechat_candidates", "strict": True, "schema": schema}},
            "max_tokens": self.settings.max_tokens,
            "stream": False,
        }
        if self.settings.provider == "openrouter":
            params["extra_body"] = {"provider": {"require_parameters": True, "allow_fallbacks": False}}
        if self.settings.temperature is not None:
            params["temperature"] = self.settings.temperature
        return params

    async def _complete(self, messages: list[dict], schema: dict, timeout: float) -> ProviderReply:
        try:
            async with AsyncOpenAI(api_key=self.settings.api_key,
                                   base_url=self.settings.base_url,
                                   max_retries=0, timeout=timeout) as client:
                response = await asyncio.wait_for(
                    client.chat.completions.create(messages=messages, **self.request_parameters(schema)),
                    timeout=timeout,
                )
        except (TimeoutError, APITimeoutError) as exc:
            raise ProviderError("PROVIDER_TIMEOUT", "Provider exceeded the attempt deadline", retryable=True) from exc
        except APIStatusError as exc:
            status = exc.status_code
            # Never serialize exception text, headers or response bodies: these
            # can contain account details or an echoed Authorization value.
            if status in (401, 403):
                code, retryable = "AUTH_ERROR", False
            elif status == 429:
                code, retryable = "RATE_LIMIT", True
            elif status == 408:
                code, retryable = "PROVIDER_TIMEOUT", True
            elif status in (400, 404, 422):
                code, retryable = "CONFIG_ERROR", False
            else:
                code, retryable = "PROVIDER_ERROR", status >= 500
            raise ProviderError(code, f"Provider returned HTTP {status}", retryable=retryable,
                                status_code=status, response_id=getattr(exc, "request_id", None),
                                error_category=classify_gateway_error(exc.response),
                                retry_after=parse_retry_after(exc.response.headers.get("Retry-After"))) from exc
        except APIConnectionError as exc:
            raise ProviderError("PROVIDER_ERROR", "Could not connect to provider", retryable=True) from exc
        raw = response.model_dump(mode="json")
        embedded = raw.get("error")
        if embedded:
            raise ProviderError("PROVIDER_ERROR", "Provider returned an embedded error", retryable=False)
        if not response.choices:
            return ProviderReply(None, raw, response.id, model=response.model)
        choice = response.choices[0]
        return ProviderReply(
            content=choice.message.content, raw=raw, response_id=response.id,
            usage=response.usage.model_dump(mode="json") if response.usage else None,
            model=response.model, finish_reason=choice.finish_reason,
            refusal=getattr(choice.message, "refusal", None),
        )

    def complete(self, messages: list[dict], schema: dict, timeout: float) -> ProviderReply:
        return asyncio.run(self._complete(messages, schema, timeout))
