"""Native Vertex AI generateContent transport; the application owns retries.

Only the two versioned application messages become model instructions/content.
Credentials, account metadata and authenticated URLs never enter request snapshots.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from typing import Any

import httpx2 as httpx

from .provider import ProviderError, ProviderReply, classify_gateway_error, parse_retry_after
from .settings import Settings

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
_BLOCKED_FINISH_REASONS = {
    "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII",
    "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION",
}


def _malformed_response() -> ProviderError:
    return ProviderError("PROVIDER_ERROR", "Provider returned an invalid generateContent response")


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _malformed_response()
    return value or None


def _token_count(metadata: dict, key: str) -> int | None:
    value = metadata.get(key)
    if value is not None and (type(value) is not int or value < 0):
        raise _malformed_response()
    return value


def normalize_vertex_response(raw: dict[str, Any]) -> ProviderReply:
    """Normalize native fields without modifying or fabricating the raw evidence."""
    if not isinstance(raw, dict):
        raise _malformed_response()
    try:
        json.dumps(raw, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _malformed_response() from exc
    if raw.get("error"):
        raise ProviderError("PROVIDER_ERROR", "Provider returned an embedded error")

    response_id = _optional_string(raw.get("responseId"))
    model = _optional_string(raw.get("modelVersion"))
    usage = None
    metadata = raw.get("usageMetadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise _malformed_response()
        prompt = _token_count(metadata, "promptTokenCount")
        output = _token_count(metadata, "candidatesTokenCount")
        thoughts = _token_count(metadata, "thoughtsTokenCount")
        total = _token_count(metadata, "totalTokenCount")
        cached = _token_count(metadata, "cachedContentTokenCount")
        usage = {}
        if prompt is not None:
            usage["prompt_tokens"] = prompt
        if output is not None or thoughts is not None:
            # Google bills thinking as output; keep it in the comparable total.
            usage["completion_tokens"] = (output or 0) + (thoughts or 0)
        if total is not None:
            usage["total_tokens"] = total
        if thoughts is not None:
            usage["completion_tokens_details"] = {"reasoning_tokens": thoughts}
        if cached is not None:
            usage["prompt_tokens_details"] = {"cached_tokens": cached}
        usage = usage or None

    feedback = raw.get("promptFeedback")
    if feedback is not None and not isinstance(feedback, dict):
        raise _malformed_response()
    block_reason = feedback.get("blockReason") if feedback else None
    if block_reason and block_reason != "BLOCK_REASON_UNSPECIFIED":
        return ProviderReply(None, raw, response_id, usage, model, "content_filter",
                             "Provider blocked this request")

    candidates = raw.get("candidates", [])
    if not isinstance(candidates, list):
        raise _malformed_response()
    if not candidates:
        return ProviderReply(None, raw, response_id, usage, model, None)
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise _malformed_response()
    reason = candidate.get("finishReason")
    if not isinstance(reason, str):
        raise _malformed_response()
    refusal = None
    if reason in _BLOCKED_FINISH_REASONS:
        finish_reason = "content_filter"
        refusal = "Provider blocked this request"
    elif reason == "MAX_TOKENS":
        finish_reason = "length"
    elif reason == "STOP":
        finish_reason = "stop"
    else:
        # An unspecified/unknown stop is not evidence of a completed response.
        raise _malformed_response()

    content = candidate.get("content")
    text = None
    if content is not None:
        if not isinstance(content, dict) or not isinstance(content.get("parts", []), list):
            raise _malformed_response()
        texts = []
        for part in content.get("parts", []):
            if not isinstance(part, dict):
                raise _malformed_response()
            if "text" in part:
                if not isinstance(part["text"], str):
                    raise _malformed_response()
                if part.get("thought") is not True:
                    texts.append(part["text"])
        text = "".join(texts) if texts else None
    return ProviderReply(text, raw, response_id, usage, model, finish_reason, refusal)


def _new_http_client(timeout: float):
    return httpx.AsyncClient(timeout=timeout, follow_redirects=False,
                             transport=httpx.AsyncHTTPTransport(retries=0))


def _default_credentials(request, credentials_path=None):
    import google.auth

    if credentials_path is not None:
        credentials, _ = google.auth.load_credentials_from_file(
            str(credentials_path), scopes=_SCOPES, request=request,
        )
        return credentials
    credentials, _ = google.auth.default(scopes=_SCOPES, request=request)
    return credentials


class _BoundedAuthRequest:
    """Cap every ADC discovery/refresh HTTP request at the attempt deadline."""

    def __init__(self, request, deadline: float):
        self._request = request
        self._deadline = deadline

    def __call__(self, *args, **kwargs):
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Authentication deadline exceeded")
        requested = kwargs.pop("timeout", None)
        if isinstance(requested, (int, float)) and math.isfinite(requested) and requested > 0:
            remaining = min(remaining, requested)
        return self._request(*args, timeout=remaining, **kwargs)


def _adc_headers(provider, deadline: float) -> dict[str, str]:
    # Imports are lazy: API-key mode needs no credential SDK or local ADC files.
    import requests
    from google.auth.transport.requests import Request

    with requests.Session() as session:
        session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))
        session.mount("http://", requests.adapters.HTTPAdapter(max_retries=0))
        request = _BoundedAuthRequest(Request(session=session), deadline)
        if provider._credentials is None:
            provider._credentials = _default_credentials(request, provider.settings.vertex_credentials_path)
        credentials = provider._credentials
        if not credentials.valid:
            credentials.refresh(request)
        token = credentials.token
        if not isinstance(token, str) or not token.strip():
            raise ValueError("ADC did not return an access token")
        headers = {"Authorization": f"Bearer {token}"}
        quota_project = getattr(credentials, "quota_project_id", None)
        if quota_project:
            headers["x-goog-user-project"] = quota_project
        return headers


class VertexAIProvider:
    source = "live"

    def __init__(self, settings: Settings):
        if settings.provider != "vertex":
            raise ProviderError("CONFIG_ERROR", "Unsupported LLM_PROVIDER for Vertex AI")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", settings.model):
            raise ProviderError("CONFIG_ERROR", "LLM_MODEL must be a bare Google model ID")
        if settings.vertex_auth_mode not in {"api_key", "adc"}:
            raise ProviderError("CONFIG_ERROR", "VERTEX_AUTH_MODE must be api_key or adc")
        if not re.fullmatch(r"[a-z][a-z0-9-]*", settings.vertex_location):
            raise ProviderError("CONFIG_ERROR", "VERTEX_LOCATION must be a Google Cloud location")
        project = settings.vertex_project or ""
        if project and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", project):
            raise ProviderError("CONFIG_ERROR", "VERTEX_PROJECT must be a Google Cloud project ID")
        if settings.vertex_auth_mode == "adc" and not project:
            raise ProviderError("CONFIG_ERROR", "VERTEX_PROJECT is required for ADC authentication")
        if settings.vertex_auth_mode == "api_key" and not settings.api_key.strip():
            raise ProviderError("CONFIG_ERROR", "Set the configured Google Cloud API key")
        if not project and settings.vertex_location != "global":
            raise ProviderError("CONFIG_ERROR", "Projectless express mode requires VERTEX_LOCATION=global")
        self.settings = settings
        self._credentials = None
        host = ("aiplatform.googleapis.com" if settings.vertex_location == "global"
                else f"{settings.vertex_location}-aiplatform.googleapis.com")
        prefix = f"projects/{project}/locations/{settings.vertex_location}/" if project else ""
        self._endpoint = (f"https://{host}/v1/{prefix}publishers/google/models/"
                          f"{settings.model}:generateContent")

    def request_parameters(self, schema: dict) -> dict:
        config = {"responseMimeType": "application/json", "responseJsonSchema": schema,
                  "maxOutputTokens": self.settings.max_tokens}
        if self.settings.temperature is not None:
            config["temperature"] = self.settings.temperature
        return {"model": self.settings.model, "generationConfig": config}

    def _request_body(self, messages: list[dict], schema: dict) -> dict:
        if (len(messages) != 2 or not all(isinstance(item, dict) for item in messages)
                or messages[0].get("role") != "system" or messages[1].get("role") != "user"
                or not all(isinstance(item.get("content"), str) for item in messages)):
            raise ProviderError("CONFIG_ERROR", "Vertex AI requires the application system and user messages")
        return {
            "systemInstruction": {"parts": [{"text": messages[0]["content"]}]},
            "contents": [{"role": "user", "parts": [{"text": messages[1]["content"]}]}],
            "generationConfig": self.request_parameters(schema)["generationConfig"],
        }

    async def _complete(self, body: dict, headers: dict, timeout: float) -> ProviderReply:
        try:
            async with _new_http_client(timeout) as client:
                response = await asyncio.wait_for(
                    client.post(self._endpoint, headers=headers, json=body), timeout=timeout,
                )
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise ProviderError("PROVIDER_TIMEOUT", "Provider exceeded the attempt deadline",
                                retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ProviderError("PROVIDER_ERROR", "Could not connect to provider", retryable=True) from exc
        if not 200 <= response.status_code < 300:
            status = response.status_code
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
                                status_code=status, error_category=classify_gateway_error(response),
                                retry_after=parse_retry_after(response.headers.get("Retry-After")))
        try:
            raw = response.json()
        except (ValueError, UnicodeError) as exc:
            raise _malformed_response() from exc
        return normalize_vertex_response(raw)

    def complete(self, messages: list[dict], schema: dict, timeout: float) -> ProviderReply:
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ProviderError("CONFIG_ERROR", "Provider timeout must be finite and positive")
        body = self._request_body(messages, schema)
        deadline = time.monotonic() + timeout
        if self.settings.vertex_auth_mode == "api_key":
            headers = {"x-goog-api-key": self.settings.api_key}
        else:
            try:
                headers = _adc_headers(self, deadline)
            except ImportError as exc:
                raise ProviderError("CONFIG_ERROR", "ADC requires google-auth and requests") from exc
            except Exception as exc:
                if time.monotonic() >= deadline or isinstance(exc, TimeoutError):
                    raise ProviderError("PROVIDER_TIMEOUT", "Provider exceeded the attempt deadline",
                                        retryable=True) from exc
                raise ProviderError("AUTH_ERROR", "Could not obtain Google Cloud ADC credentials") from exc
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProviderError("PROVIDER_TIMEOUT", "Provider exceeded the attempt deadline", retryable=True)
        return asyncio.run(self._complete(body, headers, remaining))
