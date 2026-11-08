"""Configuration stays outside business inputs and never enters run logs."""

from dataclasses import dataclass, field
from pathlib import Path
import math
import os
import ipaddress
import re
import hashlib
from urllib.parse import unquote, urlsplit

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROVIDER_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "tokenrouter": "https://api.tokenrouter.com/v1",
}
VERTEX_DEFAULT_MODEL = "gemini-3.1-flash-lite"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_DEFAULT_MODEL = "gemini-3.1-flash-lite"


def _vertex_service_url(location: str) -> str:
    if location == "global":
        return "https://aiplatform.googleapis.com/v1"
    if not re.fullmatch(r"[a-z]+(?:-[a-z0-9]+)+[0-9]", location):
        raise ValueError("GOOGLE_CLOUD_LOCATION must be global or a physical region such as us-central1; us/eu multi-region routing is not supported")
    return f"https://{location}-aiplatform.googleapis.com/v1"


def _validated_base_url(value: str) -> str:
    if not isinstance(value, str) or not value or any(c.isspace() for c in value) or "\\" in value:
        raise ValueError("LLM_BASE_URL must be a valid HTTP(S) base URL without whitespace")
    try:
        parts = urlsplit(value)
        hostname = parts.hostname
        port = parts.port
    except ValueError as exc:
        raise ValueError("LLM_BASE_URL has an invalid host or port") from exc
    if parts.scheme not in ("https", "http") or not hostname:
        raise ValueError("LLM_BASE_URL must use http:// or https:// and include a host")
    if parts.username is not None or parts.password is not None or "?" in value or "#" in value:
        raise ValueError("LLM_BASE_URL must not contain credentials, a query, or a fragment")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("LLM_BASE_URL has an invalid port")
    try:
        if ":" in hostname:
            ipaddress.IPv6Address(hostname)
        else:
            host_ascii = hostname.encode("idna").decode("ascii")
            if len(host_ascii) > 253 or not re.fullmatch(
                r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*\.?",
                host_ascii,
            ):
                raise ValueError("invalid host")
    except (UnicodeError, ValueError) as exc:
        raise ValueError("LLM_BASE_URL has an invalid host") from exc
    if unquote(parts.path).rstrip("/").lower().endswith("/chat/completions"):
        raise ValueError("LLM_BASE_URL must be the API base URL, without /chat/completions")
    return value.rstrip("/")


@dataclass(frozen=True)
class Settings:
    runs_dir: Path = Path("runs")
    brand_path: Path = PROJECT_ROOT / "config" / "brand.json"
    provider: str = "openrouter"
    model: str = ""
    api_key: str = field(default="", repr=False)
    base_url: str | None = None
    max_attempts: int = 3
    timeout_seconds: float = 30.0
    total_timeout_seconds: float = 90.0
    max_input_bytes: int = 131072
    max_tokens: int = 1200
    temperature: float | None = None
    max_total_calls: int | None = None
    vertex_project: str = field(default="", repr=False)
    vertex_location: str = "global"
    vertex_auth_mode: str = "adc"
    vertex_credentials_path: Path | None = field(default=None, repr=False)

    def __post_init__(self):
        provider = self.provider.strip().lower()
        if provider not in (*PROVIDER_BASE_URLS, "openai_compatible", "vertex", "gemini"):
            raise ValueError("LLM_PROVIDER must be openrouter, tokenrouter, openai_compatible, vertex, or gemini")
        object.__setattr__(self, "provider", provider)
        if self.vertex_credentials_path is not None:
            credentials_path = Path(self.vertex_credentials_path)
            if not credentials_path.is_absolute():
                credentials_path = PROJECT_ROOT / credentials_path
            object.__setattr__(self, "vertex_credentials_path", credentials_path.resolve())
        if provider == "vertex":
            if self.vertex_project and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", self.vertex_project):
                raise ValueError("GOOGLE_CLOUD_PROJECT must be a project identifier without path or query characters")
            auth_mode = self.vertex_auth_mode.strip().lower()
            if auth_mode not in ("api_key", "adc"):
                raise ValueError("VERTEX_AUTH_MODE must be api_key or adc")
            object.__setattr__(self, "vertex_auth_mode", auth_mode)
            expected_base = _vertex_service_url(self.vertex_location)
            base_url = self.base_url or expected_base
            if _validated_base_url(base_url) != expected_base:
                raise ValueError("Vertex LLM_BASE_URL must match the Google API host and /v1 path for GOOGLE_CLOUD_LOCATION")
            model = self.model or VERTEX_DEFAULT_MODEL
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", model):
                raise ValueError("Vertex LLM_MODEL must be a bare model ID without path or query characters")
            object.__setattr__(self, "model", model)
        elif provider == "gemini":
            base_url = self.base_url or GEMINI_BASE_URL
            if _validated_base_url(base_url) != GEMINI_BASE_URL:
                raise ValueError("Gemini LLM_BASE_URL must be https://generativelanguage.googleapis.com/v1beta")
            model = self.model or GEMINI_DEFAULT_MODEL
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", model):
                raise ValueError("Gemini LLM_MODEL must be a bare model ID without path or query characters")
            object.__setattr__(self, "model", model)
        else:
            base_url = self.base_url or PROVIDER_BASE_URLS.get(provider)
        if base_url is None:
            raise ValueError("LLM_BASE_URL is required for LLM_PROVIDER=openai_compatible")
        object.__setattr__(self, "base_url", _validated_base_url(base_url))
        if not 1 <= self.max_attempts <= 3:
            raise ValueError("LLM_MAX_ATTEMPTS must be between 1 and 3")
        for name in ("timeout_seconds", "total_timeout_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_input_bytes < 1024 or self.max_tokens < 100:
            raise ValueError("Input/token limits are too small")
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ValueError("LLM_TEMPERATURE must be between 0 and 2")
        if self.max_total_calls is not None and self.max_total_calls < 1:
            raise ValueError("LLM_MAX_TOTAL_CALLS must be positive")

    @property
    def route_identity(self) -> dict[str, str | None]:
        """Non-secret routing identity for run provenance and batch resumption."""
        identity = {"provider": self.provider, "base_url": self.base_url}
        if self.provider == "vertex":
            identity.update({
                "vertex_location": self.vertex_location,
                "vertex_auth_mode": self.vertex_auth_mode,
                "vertex_project_sha256": hashlib.sha256(self.vertex_project.encode("utf-8")).hexdigest() if self.vertex_project else None,
            })
        return identity

    @classmethod
    def from_env(cls, env_file: Path | None = None):
        values = {**dotenv_values(env_file or PROJECT_ROOT / ".env"), **os.environ}
        provider = (values.get("LLM_PROVIDER") or "openrouter").strip().lower()
        google_api_key = values.get("GOOGLE_CLOUD_API_KEY") or values.get("GOOGLE_API_KEY") or ""
        if provider == "openrouter":
            # Preserve the original OpenRouter alias priority for existing .env files.
            api_key = values.get("OPENROUTER_API_KEY") or values.get("LLM_API_KEY") or ""
        elif provider == "tokenrouter":
            api_key = values.get("LLM_API_KEY") or values.get("TOKENROUTER_API_KEY") or ""
        elif provider == "vertex":
            api_key = google_api_key
        elif provider == "gemini":
            api_key = values.get("GEMINI_API_KEY") or values.get("GOOGLE_API_KEY") or ""
        else:
            api_key = values.get("LLM_API_KEY") or ""
        return cls(
            runs_dir=Path(values.get("RUNS_DIR", "runs")),
            brand_path=Path(values.get("BRAND_PATH", str(PROJECT_ROOT / "config" / "brand.json"))),
            provider=provider,
            model=values.get("LLM_MODEL") or (VERTEX_DEFAULT_MODEL if provider == "vertex" else GEMINI_DEFAULT_MODEL if provider == "gemini" else ""),
            api_key=api_key,
            base_url=values.get("LLM_BASE_URL") or None,
            max_attempts=int(values.get("LLM_MAX_ATTEMPTS", "3")),
            timeout_seconds=float(values.get("LLM_TIMEOUT_SECONDS", "30")),
            total_timeout_seconds=float(values.get("LLM_TOTAL_TIMEOUT_SECONDS", "90")),
            max_input_bytes=int(values.get("MAX_INPUT_BYTES", "131072")),
            max_tokens=int(values.get("LLM_MAX_TOKENS", "1200")),
            temperature=float(values["LLM_TEMPERATURE"]) if values.get("LLM_TEMPERATURE") else None,
            max_total_calls=int(values["LLM_MAX_TOTAL_CALLS"]) if values.get("LLM_MAX_TOTAL_CALLS") else None,
            vertex_project=values.get("GOOGLE_CLOUD_PROJECT") or "",
            vertex_location=values.get("GOOGLE_CLOUD_LOCATION") or "global",
            vertex_auth_mode=values.get("VERTEX_AUTH_MODE") or ("api_key" if google_api_key else "adc"),
            vertex_credentials_path=Path(values["GOOGLE_APPLICATION_CREDENTIALS"]) if values.get("GOOGLE_APPLICATION_CREDENTIALS") else None,
        )
