"""Real installed SDK parsing through an in-memory HTTP transport, never network."""

import json

import httpx2
from openai import AsyncOpenAI
import pytest

from ca_tool.prompts import OUTPUT_SCHEMA
from ca_tool.provider import OpenRouterProvider, ProviderError, classify_gateway_error, parse_retry_after
from ca_tool.settings import Settings


def install_transport(monkeypatch, handler):
    def factory(**kwargs):
        assert kwargs["max_retries"] == 0
        return AsyncOpenAI(**kwargs, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    monkeypatch.setattr("ca_tool.provider.AsyncOpenAI", factory)


def test_sdk_request_and_response_contract(monkeypatch):
    captured = []
    def handler(request):
        captured.append(json.loads(request.content))
        assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
        return httpx2.Response(200, json={
            "id": "fixture-response", "object": "chat.completion", "created": 1,
            "model": "fixture/model", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": '{"candidates":[]}'}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
        })
    install_transport(monkeypatch, handler)
    provider = OpenRouterProvider(Settings(api_key="fictional-test-key", model="fixture/model"))
    reply = provider.complete([{"role": "user", "content": "test fixture"}], OUTPUT_SCHEMA, 1)
    assert reply.response_id == "fixture-response"
    assert reply.usage["total_tokens"] == 11
    assert captured[0]["provider"]["require_parameters"] is True
    assert captured[0]["provider"]["allow_fallbacks"] is False
    assert captured[0]["response_format"]["json_schema"]["strict"] is True
    assert "temperature" not in captured[0]


@pytest.mark.parametrize("status,code,retryable", [(401,"AUTH_ERROR",False), (403,"AUTH_ERROR",False),
    (429,"RATE_LIMIT",True), (500,"PROVIDER_ERROR",True), (400,"CONFIG_ERROR",False),
    (402,"PROVIDER_ERROR",False), (408,"PROVIDER_TIMEOUT",True)])
def test_sdk_http_failures_are_sanitized_without_sdk_retry(monkeypatch, status, code, retryable):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx2.Response(status, headers={"Retry-After": "2"},
                               json={"error": {"message": "secret fictional-test-key", "code": status}})
    install_transport(monkeypatch, handler)
    with pytest.raises(ProviderError) as error:
        OpenRouterProvider(Settings(api_key="fictional-test-key", model="fixture/model")).complete([], OUTPUT_SCHEMA, 1)
    assert error.value.code == code
    assert error.value.retryable is retryable
    assert error.value.retry_after == 2
    assert "fictional-test-key" not in str(error.value.as_dict())
    assert len(calls) == 1


def test_retry_after_date_and_bad_values():
    assert parse_retry_after("2.5") == 2.5
    assert parse_retry_after("Tue, 01 Jan 2030 12:00:00 GMT") >= 0
    assert parse_retry_after("invalid") is None
    assert parse_retry_after("inf") is None


@pytest.mark.parametrize("provider_name,base_url", [
    ("tokenrouter", "https://api.tokenrouter.com/v1"),
    ("openai_compatible", "https://fictional-gateway.example/v1"),
])
def test_custom_gateway_uses_configured_url_without_openrouter_extensions(monkeypatch, provider_name, base_url):
    calls = []

    def handler(request):
        calls.append(request)
        assert str(request.url) == base_url + "/chat/completions"
        body = json.loads(request.content)
        assert "provider" not in body
        assert body["response_format"]["json_schema"]["strict"] is True
        assert body["model"] == "fixture/model"
        return httpx2.Response(200, json={
            "id": "fixture-gateway-response", "object": "chat.completion", "created": 1,
            "model": "fixture/model", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": '{"candidates":[]}'}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
        })

    install_transport(monkeypatch, handler)
    settings = Settings(provider=provider_name, base_url=base_url,
                        api_key="fictional-test-key", model="fixture/model")
    reply = OpenRouterProvider(settings).complete([], OUTPUT_SCHEMA, 1)
    assert len(calls) == 1
    assert reply.response_id == "fixture-gateway-response"


@pytest.mark.parametrize("body,category", [
    ("Invalid API key: fictional-secret", "invalid_credentials"),
    ("Model permission denied: fictional-secret", "model_access_denied"),
    ("Insufficient credit balance: fictional-secret", "insufficient_credits"),
    ("Cloudflare block: fictional-secret", "gateway_firewall"),
    ("Unspecified error: fictional-secret", None),
])
def test_error_diagnostics_are_fixed_labels_without_provider_text(body, category):
    assert classify_gateway_error(httpx2.Response(403, text=body)) == category
