"""AI Studio native JSON integration uses only fictional in-memory transport."""

from dataclasses import replace
import json
from pathlib import Path

import httpx2
import pytest

from ca_tool.batch import run_batch
from ca_tool.gemini_provider import GeminiProvider
from ca_tool.pipeline import generate
from ca_tool.prompts import OUTPUT_SCHEMA
from ca_tool.provider import create_provider
from ca_tool.settings import Settings
from ca_tool.verify_run import file_hash, verify_run


@pytest.fixture
def gemini_run(tmp_path, monkeypatch):
    raw_input = {
        "request_id": "GEMINI-OFFLINE-INTEGRATION", "brand_id": "lanxu", "scenario": "product_recommendation",
        "reference_time": "2026-11-03T14:00:00+08:00",
        "client": {"id": "fictional-gemini-test", "contact_permission": "allowed"},
        "conversation_history": [{"speaker": "client", "content": "我想先了解肩背款式的资料。"}],
        "products": [{"id": "LX-P001", "name": "云影肩包", "price": 18000, "currency": "CNY", "carry_options": ["肩背"]}],
        "scenario_context": {},
    }
    candidates = [
        {"id": "option_1", "text": "您好，云影肩包支持肩背，正式价格是18000元。您可以先了解这些资料，再按自己的需要考虑。"},
        {"id": "option_2", "text": "想和您分享云影肩包的已知信息。如果您愿意，也可以告诉我希望进一步了解哪些使用细节。"},
        {"id": "option_3", "text": "这款云影肩包可以先作为一个了解的方向。您不必急着决定，也欢迎说说日常的使用需要。"},
    ]
    text = json.dumps({"candidates": candidates}, ensure_ascii=False, separators=(",", ":"))
    cut = len(text) // 2
    native = {
        "responseId": "gemini-fixture-not-live", "modelVersion": "gemini-3.1-flash-lite",
        "candidates": [{"index": 0, "content": {"role": "model", "parts": [
            {"text": text[:cut]}, {"text": text[cut:]},
        ]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 40, "thoughtsTokenCount": 4, "totalTokenCount": 56},
    }
    requests = []

    def never_authenticate_with_cloud(*args, **kwargs):
        raise AssertionError("Gemini Developer API attempted Cloud ADC authentication")

    def handler(request):
        requests.append(request)
        assert request.method == "POST"
        assert str(request.url) == "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent"
        assert request.headers["x-goog-api-key"] == "fictional-gemini-transport-key"
        assert "Authorization" not in request.headers
        assert "x-goog-user-project" not in request.headers
        body = json.loads(request.content)
        assert body["generationConfig"]["responseJsonSchema"] == OUTPUT_SCHEMA
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        assert "systemInstruction" in body and len(body["contents"]) == 1
        assert "projects/" not in str(request.url)
        return httpx2.Response(200, json=native)

    monkeypatch.setattr("ca_tool.vertex_provider._adc_headers", never_authenticate_with_cloud)
    monkeypatch.setattr("ca_tool.vertex_provider._new_http_client", lambda timeout: httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler), timeout=timeout, follow_redirects=False))
    settings = Settings(provider="gemini", api_key="fictional-gemini-transport-key", runs_dir=tmp_path / "runs",
                        vertex_project="fictional-unused-cloud-project", vertex_auth_mode="adc", vertex_location="us-central1",
                        vertex_credentials_path=tmp_path / "fictional-unused-cloud-credentials.json")

    def offline_generate(raw, configured):
        adapter = create_provider(configured)
        assert isinstance(adapter, GeminiProvider)
        adapter.source = "mock"  # Never make a live-evidence claim for HTTP fixtures.
        return generate(raw, configured, provider=adapter)

    result = offline_generate(raw_input, settings)
    assert result["status"] == "success", result
    return {"input": raw_input, "result": result, "run_dir": Path(result["metadata"]["run_dir"]),
            "native": native, "candidates": candidates, "requests": requests, "settings": settings,
            "offline_generate": offline_generate}


def _change_attempt(run_dir, mutate):
    path = run_dir / "attempt_01.json"
    attempt = json.loads(path.read_text(encoding="utf-8"))
    mutate(attempt)
    path.write_text(json.dumps(attempt, ensure_ascii=False), encoding="utf-8")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][path.name] = file_hash(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_gemini_factory_pipeline_preserves_native_reply_and_stays_mock(gemini_run):
    result = gemini_run["result"]
    assert result["metadata"]["source"] == "mock"
    assert result["candidates"] == gemini_run["candidates"]
    assert result["metadata"]["usage"] == {"prompt_tokens": 12, "completion_tokens": 44, "total_tokens": 56}
    assert len(gemini_run["requests"]) == 1
    attempt = json.loads((gemini_run["run_dir"] / "attempt_01.json").read_text(encoding="utf-8"))
    assert attempt["response"]["raw"] == gemini_run["native"]
    assert verify_run(gemini_run["run_dir"]) == []
    assert any("expected actual live" in error for error in verify_run(gemini_run["run_dir"], expect_live=True))


@pytest.mark.parametrize("field,value", [
    ("content", '{"candidates":[]}'),
    ("refusal", "fictional-changed-refusal"),
    ("finish_reason", "length"),
    ("response_id", "fictional-changed-id"),
    ("model", "fictional-changed-model"),
    ("usage", {"prompt_tokens": 999, "completion_tokens": 44, "total_tokens": 1043}),
])
def test_gemini_normalized_field_tampering_is_detected_after_rehash(gemini_run, field, value):
    _change_attempt(gemini_run["run_dir"], lambda attempt: attempt["response"].update({field: value}))
    errors = verify_run(gemini_run["run_dir"])
    assert not any("file hash mismatch" in error for error in errors)
    assert any(f"normalized Gemini {field} differs" in error for error in errors)


def test_gemini_native_response_tampering_is_detected(gemini_run):
    _change_attempt(gemini_run["run_dir"], lambda attempt: attempt["response"]["raw"]["usageMetadata"].update(totalTokenCount=999))
    assert any("normalized Gemini usage differs" in error for error in verify_run(gemini_run["run_dir"]))


@pytest.mark.parametrize("field,value", [
    ("provider", "vertex"),
    ("base_url", "https://aiplatform.googleapis.com/v1"),
    ("base_url", None),
])
def test_gemini_route_changes_or_missing_fields_are_detected(gemini_run, field, value):
    def mutate(attempt):
        if value is None:
            attempt["request"].pop(field)
        else:
            attempt["request"][field] = value
    _change_attempt(gemini_run["run_dir"], mutate)
    assert any(f"Gemini route {field}" in error for error in verify_run(gemini_run["run_dir"]))


@pytest.mark.parametrize("field", [
    "vertex_project", "vertex_location", "vertex_auth_mode", "vertex_project_sha256", "vertex_credentials_path",
    "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "GOOGLE_APPLICATION_CREDENTIALS", "VERTEX_AUTH_MODE",
    "GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_API_KEY", "api_key", "credentials_path",
    "Authorization", "authorization", "x-goog-api-key",
])
def test_gemini_snapshot_rejects_cloud_and_private_credential_fields(gemini_run, field):
    _change_attempt(gemini_run["run_dir"], lambda attempt: attempt["request"].update({field: "fictional-private-sentinel"}))
    assert any("Gemini snapshots must omit Cloud routing and credentials" in error for error in verify_run(gemini_run["run_dir"]))


def test_gemini_snapshots_exclude_cloud_config_key_and_credential_path(gemini_run):
    settings = gemini_run["settings"]
    snapshots = "\n".join(path.read_text(encoding="utf-8") for path in gemini_run["run_dir"].glob("*.json"))
    assert settings.api_key not in snapshots
    assert settings.vertex_project not in snapshots
    assert "fictional-unused-cloud-credentials" not in snapshots
    assert str(settings.vertex_credentials_path) not in snapshots
    metadata = gemini_run["result"]["metadata"]
    assert not any(key.startswith("vertex_") for key in metadata)
    assert settings.route_identity == {"provider": "gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta"}


@pytest.mark.parametrize("sampling_change", [None, {"max_tokens": 1300}, {"temperature": 0.4}])
def test_gemini_batch_resume_checks_native_generation_config_without_new_calls(gemini_run, tmp_path, monkeypatch, sampling_change):
    dataset = tmp_path / "gemini-cases.jsonl"
    dataset.write_text(json.dumps(gemini_run["input"], ensure_ascii=False) + "\n", encoding="utf-8")
    output = tmp_path / "initial.csv"
    settings = gemini_run["settings"]
    manifest = run_batch(dataset, output, settings, max_total_calls=1, generate_fn=gemini_run["offline_generate"])
    assert manifest["entries"][0]["source"] == "mock"
    assert manifest["new_provider_calls"] == 1
    request_count = len(gemini_run["requests"])
    # Simulate only live eligibility to exercise the production resume gate;
    # underlying fixture records remain source=mock and are verified normally.
    manifest["entries"][0]["source"] = "live"
    prior = tmp_path / "mock-resume-eligibility-fixture.json"
    prior.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr("ca_tool.batch.verify_run", lambda path, expect_live: verify_run(path, expect_live=False))

    def must_not_generate(*args, **kwargs):
        raise AssertionError("resume unexpectedly invoked a new model call")

    if sampling_change is not None:
        with pytest.raises(ValueError, match="resume sampling parameters changed"):
            run_batch(dataset, tmp_path / "resumed.csv", replace(settings, **sampling_change), max_total_calls=1,
                      resume_manifest=prior, generate_fn=must_not_generate)
    else:
        resumed = run_batch(dataset, tmp_path / "resumed.csv", settings, max_total_calls=1,
                            resume_manifest=prior, generate_fn=must_not_generate)
        assert resumed["new_provider_calls"] == 0
        assert resumed["entries"][0]["reused"] is True
        assert resumed["entries"][0]["source"] == "mock"
    assert len(gemini_run["requests"]) == request_count
