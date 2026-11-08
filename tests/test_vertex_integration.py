"""Native Google fixtures through the real pipeline, with no network or live claim."""

import json
from pathlib import Path

import httpx2
import pytest

from ca_tool.pipeline import generate
from ca_tool.prompts import OUTPUT_SCHEMA
from ca_tool.provider import OpenRouterProvider, create_provider
from ca_tool.settings import PROJECT_ROOT, Settings
from ca_tool.verify_run import file_hash, verify_run
from ca_tool.vertex_provider import VertexAIProvider


@pytest.fixture
def native_run(tmp_path, monkeypatch):
    raw_input = {
        "request_id": "VERTEX-OFFLINE-INTEGRATION", "brand_id": "lanxu", "scenario": "product_recommendation",
        "reference_time": "2026-11-03T14:00:00+08:00",
        "client": {"id": "fictional-vertex-test", "contact_permission": "allowed"},
        "conversation_history": [{"speaker": "client", "content": "我想先了解商品的实际资料。"}],
        "products": [{"id": "LX-P001", "name": "云影肩包", "price": 18000, "currency": "CNY", "carry_options": ["肩背"]}],
        "scenario_context": {},
    }
    candidates = {"candidates": [
        {"id": "option_1", "text": "您好，云影肩包支持肩背，正式价格是18000元。您可以先了解这些资料，再按自己的需要考虑。"},
        {"id": "option_2", "text": "想和您分享云影肩包的已知信息。如果您愿意，也可以告诉我希望进一步了解哪些使用细节。"},
        {"id": "option_3", "text": "这款云影肩包可以先作为一个了解的方向。您不必急着决定，也欢迎说说日常的使用需要。"},
    ]}
    text = json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))
    split = len(text) // 2
    google_raw = {
        "responseId": "fixture-google-response-not-live", "modelVersion": "gemini-3.1-flash-lite",
        "candidates": [{"index": 0, "content": {"role": "model", "parts": [
            {"text": text[:split]}, {"text": text[split:]},
        ]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 40, "thoughtsTokenCount": 4, "totalTokenCount": 56},
    }
    requests = []

    def handler(request):
        requests.append(request)
        assert request.method == "POST"
        assert str(request.url) == "https://aiplatform.googleapis.com/v1/projects/fictional-private-google-project/locations/global/publishers/google/models/gemini-3.1-flash-lite:generateContent"
        assert request.headers["x-goog-api-key"] == "fictional-google-secret-for-transport-only"
        body = json.loads(request.content)
        schema = body["generationConfig"]["responseJsonSchema"]
        assert schema == OUTPUT_SCHEMA
        assert schema["properties"]["candidates"]["minItems"] == schema["properties"]["candidates"]["maxItems"] == 3
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        assert "systemInstruction" in body and len(body["contents"]) == 1
        return httpx2.Response(200, json=google_raw)

    monkeypatch.setattr("ca_tool.vertex_provider._new_http_client", lambda timeout: httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler), timeout=timeout, follow_redirects=False))
    settings = Settings(provider="vertex", api_key="fictional-google-secret-for-transport-only",
                        vertex_auth_mode="api_key", vertex_project="fictional-private-google-project",
                        vertex_credentials_path=tmp_path / "fictional-private-adc-file.json",
                        brand_path=PROJECT_ROOT / "config" / "brand.json", runs_dir=tmp_path / "runs")
    adapter = create_provider(settings)
    assert isinstance(adapter, VertexAIProvider)
    adapter.source = "mock"  # Keep offline transport tests out of all live evidence.
    result = generate(raw_input, settings, provider=adapter)
    assert result["status"] == "success", result
    return {"result": result, "run_dir": Path(result["metadata"]["run_dir"]), "raw": google_raw,
            "candidates": candidates["candidates"], "requests": requests, "settings": settings}


def _rewrite_attempt(run_dir, mutate):
    path = run_dir / "attempt_01.json"
    attempt = json.loads(path.read_text(encoding="utf-8"))
    mutate(attempt)
    path.write_text(json.dumps(attempt, ensure_ascii=False), encoding="utf-8")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][path.name] = file_hash(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_native_factory_pipeline_and_raw_google_provenance(native_run):
    result, run_dir = native_run["result"], native_run["run_dir"]
    assert result["metadata"]["source"] == "mock"
    assert len(native_run["requests"]) == 1
    assert result["candidates"] == native_run["candidates"]
    assert result["metadata"]["usage"] == {"prompt_tokens": 12, "completion_tokens": 44, "total_tokens": 56}
    attempt = json.loads((run_dir / "attempt_01.json").read_text(encoding="utf-8"))
    assert attempt["response"]["raw"] == native_run["raw"]
    assert attempt["response"]["content"] == "".join(part["text"] for part in native_run["raw"]["candidates"][0]["content"]["parts"])
    assert verify_run(run_dir) == []
    assert any("expected actual live" in error for error in verify_run(run_dir, expect_live=True))


def test_legacy_factory_keeps_chat_completion_adapter():
    adapter = create_provider(Settings(provider="openrouter", api_key="fictional-openrouter-key", model="fixture/model"))
    assert isinstance(adapter, OpenRouterProvider)


@pytest.mark.parametrize("field,value", [
    ("content", '{"candidates":[]}'),
    ("refusal", "fixture-refusal-not-in-original"),
    ("finish_reason", "length"),
    ("response_id", "fixture-changed-response-id"),
    ("model", "fixture-changed-model"),
    ("usage", {"prompt_tokens": 999, "completion_tokens": 44, "total_tokens": 1043}),
])
def test_native_normalization_mismatch_is_detected_after_file_hash_recomputed(native_run, field, value):
    _rewrite_attempt(native_run["run_dir"], lambda attempt: attempt["response"].update({field: value}))
    errors = verify_run(native_run["run_dir"])
    assert not any("file hash mismatch" in error for error in errors)
    assert any(f"normalized Vertex {field} differs" in error for error in errors)


def test_changed_original_google_usage_is_detected(native_run):
    _rewrite_attempt(native_run["run_dir"], lambda attempt: attempt["response"]["raw"]["usageMetadata"].update(totalTokenCount=999))
    assert any("normalized Vertex usage differs" in error for error in verify_run(native_run["run_dir"]))


def test_malformed_original_google_response_is_detected(native_run):
    _rewrite_attempt(native_run["run_dir"], lambda attempt: attempt["response"].update(raw={"fixture": "missing-native-fields"}))
    assert any("normalized Vertex" in error or "raw generateContent" in error for error in verify_run(native_run["run_dir"]))


@pytest.mark.parametrize("field,value", [
    ("provider", "openrouter"),
    ("base_url", "https://us-central1-aiplatform.googleapis.com/v1"),
    ("vertex_location", "us-central1"),
    ("vertex_auth_mode", "adc"),
    ("vertex_project_sha256", "0" * 64),
])
def test_vertex_route_mismatch_between_metadata_and_request_is_detected(native_run, field, value):
    _rewrite_attempt(native_run["run_dir"], lambda attempt: attempt["request"].update({field: value}))
    assert any(f"Vertex route {field}" in error for error in verify_run(native_run["run_dir"]))


def test_vertex_route_requires_explicit_project_hash_even_for_empty_project(native_run):
    _rewrite_attempt(native_run["run_dir"], lambda attempt: attempt["request"].pop("vertex_project_sha256"))
    assert any("Vertex route vertex_project_sha256" in error for error in verify_run(native_run["run_dir"]))


@pytest.mark.parametrize("field", ["vertex_project", "vertex_credentials_path", "api_key"])
def test_vertex_request_snapshot_rejects_private_route_fields(native_run, field):
    _rewrite_attempt(native_run["run_dir"], lambda attempt: attempt["request"].update({field: "fictional-private-value"}))
    assert any("must omit project identifiers" in error for error in verify_run(native_run["run_dir"]))


def test_vertex_snapshots_exclude_actual_project_key_and_adc_path(native_run):
    settings = native_run["settings"]
    snapshots = "\n".join(path.read_text(encoding="utf-8") for path in native_run["run_dir"].glob("*.json"))
    assert settings.api_key not in snapshots
    assert settings.vertex_project not in snapshots
    assert str(settings.vertex_credentials_path) not in snapshots
    assert "fictional-private-adc-file" not in snapshots
    assert settings.route_identity["vertex_project_sha256"] in snapshots
