"""Budget/provenance tests use labeled doubles and temporary files only."""
import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from ca_tool.batch import load_cases, preflight, run_batch
from ca_tool.evaluate import _review_rows, evaluate_manifest
from ca_tool.pipeline import generate
from ca_tool.provider import ProviderReply
from ca_tool.settings import Settings
from ca_tool.verify_run import canonical_hash, file_hash, verify_run


@pytest.fixture
def setup_cases(tmp_path):
    brand = {"brand_id": "lanxu", "name": "岚序 LANXU", "brand_rules_version": "test-only",
             "category": "皮具", "audience": "虚构成年客户", "voice": ["克制", "自然"]}
    brand_path = tmp_path / "brand.json"
    brand_path.write_text(json.dumps(brand, ensure_ascii=False), encoding="utf-8")
    raw = {"request_id": "TEST-001", "brand_id": "lanxu", "scenario": "product_recommendation",
           "reference_time": "2026-11-03T14:00:00+08:00",
           "client": {"id": "fictional-test", "contact_permission": "allowed"},
           "conversation_history": [{"speaker": "client", "content": "想了解,皮具\n先看看"}],
           "products": [{"id": "LX-P001", "name": "云影肩包", "price": 18000, "currency": "CNY"}],
           "scenario_context": {}}
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps(raw, ensure_ascii=False) + "\n", encoding="utf-8")
    return dataset, raw, Settings(brand_path=brand_path, runs_dir=tmp_path / "runs")


class LabeledDouble:
    source = "mock"

    def complete(self, messages, schema, timeout):
        content = {"candidates": [
            {"id": "option_1", "text": "您好，这款云影肩包可以作为一个选择，您平时希望了解哪些使用细节呢？"},
            {"id": "option_2", "text": "想和您分享云影肩包，若您愿意聊聊使用需要，我可以再介绍已知的商品信息。"},
            {"id": "option_3", "text": "如果您方便，可以先看看云影肩包；也欢迎告诉我您的偏好，按自己的节奏了解就好。"},
        ]}
        return ProviderReply(content=content, raw={"fixture": "unit-test-only"}, response_id="mock-unit-test")


def test_prepare_only_exports_pending_and_never_calls_provider(setup_cases, tmp_path):
    dataset, raw, settings = setup_cases
    def must_not_run(*args, **kwargs):
        raise AssertionError("prepare-only attempted provider execution")
    output = tmp_path / "artifacts" / "test_cases.csv"
    manifest = run_batch(dataset, output, settings, prepare_only=True, generate_fn=must_not_run)
    assert manifest["new_provider_calls"] == 0
    assert manifest["entries"][0]["status"] == "pending"
    assert manifest["entries"][0]["source"] == "not_run"
    assert manifest["entries"][0]["result_path"] is None
    assert manifest["entries"][0]["input_sha256"] == canonical_hash(raw)
    with output.open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["Output 1"] == row["Output 2"] == row["Output 3"] == ""
    assert json.loads(row["Conversation history"])[0]["content"] == "想了解,皮具\n先看看"
    summary = evaluate_manifest(output.parent / "final_manifest.json", output.parent / "report.md")
    assert summary["verified_live_successes"] == 0
    assert summary["verified_candidate_count"] == 0
    assert summary["reviews"]["human"]["status"] == "pending"
    assert not summary["integrity_errors"]
    assert not any(summary["case_errors"].values())
    assert not summary["is_complete_final_50"]


def test_live_batch_requires_explicit_budget(setup_cases, tmp_path):
    dataset, _, settings = setup_cases
    with pytest.raises(ValueError, match="explicit positive"):
        run_batch(dataset, tmp_path / "out.csv", settings)


def test_call_budget_covers_retry_allowance_and_retains_unexecuted_cases(setup_cases, tmp_path):
    dataset, raw, settings = setup_cases
    cases = [dict(raw, request_id=f"BUDGET-{i}") for i in range(3)]
    dataset.write_text("".join(json.dumps(value) + "\n" for value in cases), encoding="utf-8")
    calls = []
    def failure_double(value, bounded_settings):
        calls.append(bounded_settings.max_attempts)
        return {"request_id": value["request_id"], "run_id": None, "scenario": value["scenario"],
                "status": "error", "candidates": [], "metadata": {"source": "mock", "attempts": 1}}
    manifest = run_batch(dataset, tmp_path / "out.csv", settings, max_total_calls=1, generate_fn=failure_double)
    assert calls == [1]
    assert manifest["new_provider_calls"] == 1
    assert len(manifest["entries"]) == 3
    assert [entry["status"] for entry in manifest["entries"]] == ["error", "pending", "pending"]
    assert manifest["entries"][1]["error"]["code"] == "BATCH_BUDGET_EXHAUSTED"


def test_duplicate_case_ids_are_rejected_before_execution(setup_cases):
    dataset, raw, _ = setup_cases
    dataset.write_text((json.dumps(raw) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate Case ID"):
        load_cases(dataset)


def test_resume_rejects_changed_fixed_case(setup_cases, tmp_path):
    dataset, raw, settings = setup_cases
    output = tmp_path / "out.csv"
    run_batch(dataset, output, settings, prepare_only=True)
    raw["client"]["preferred_name"] = "虚构改动"
    dataset.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="resume input differs"):
        run_batch(dataset, output, settings, max_total_calls=1, resume_manifest=tmp_path / "final_manifest.json")


def _resume_routing_fixture(setup_cases, tmp_path, monkeypatch):
    """Exercise resume routing after simulated eligibility, never live execution.

    Run contents remain explicitly source=mock. Only the temporary batch entry
    and verifier eligibility are overridden to reach the protected resume branch;
    this fixture is not evidence of a real supplier response.
    """
    dataset, _, settings = setup_cases
    settings = replace(settings, model="fixture/model")

    class ResumeDouble(LabeledDouble):
        def request_parameters(self, schema):
            return {"max_tokens": settings.max_tokens, "temperature": settings.temperature}

    def fake_generate(raw, bounded):
        return generate(raw, bounded, provider=ResumeDouble())

    manifest = run_batch(dataset, tmp_path / "initial.csv", settings, max_total_calls=3, generate_fn=fake_generate)
    entry = manifest["entries"][0]
    result_path = Path(entry["result_path"])
    assert json.loads(result_path.read_text(encoding="utf-8"))["metadata"]["source"] == "mock"
    entry["source"] = "live"  # Simulate eligibility; retained run is still labeled mock.
    prior_path = tmp_path / "routing_resume_fixture.json"
    prior_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr("ca_tool.batch.verify_run", lambda *args, **kwargs: [])
    return dataset, settings, prior_path, manifest, result_path


def test_resume_with_same_provider_and_base_reuses_without_calls(setup_cases, tmp_path, monkeypatch):
    dataset, settings, prior, _, _ = _resume_routing_fixture(setup_cases, tmp_path, monkeypatch)

    def must_not_generate(*args, **kwargs):
        raise AssertionError("matching resume attempted another provider call")

    result = run_batch(dataset, tmp_path / "resumed.csv", settings, max_total_calls=1,
                       resume_manifest=prior, generate_fn=must_not_generate)
    assert result["new_provider_calls"] == 0
    assert result["entries"][0]["reused"] is True
    assert result["entries"][0]["source"] == "mock"


@pytest.mark.parametrize("change", [
    {"provider": "tokenrouter"},
    {"base_url": "https://other-gateway.example/v1"},
])
def test_resume_rejects_changed_gateway_even_with_same_model_and_source(setup_cases, tmp_path, monkeypatch, change):
    dataset, settings, prior, _, _ = _resume_routing_fixture(setup_cases, tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="resume provider/base URL changed"):
        run_batch(dataset, tmp_path / "resumed.csv", replace(settings, **change),
                  max_total_calls=1, resume_manifest=prior)


@pytest.mark.parametrize("field", ["provider", "base_url"])
def test_resume_rejects_missing_gateway_provenance(setup_cases, tmp_path, monkeypatch, field):
    dataset, settings, prior, manifest, result_path = _resume_routing_fixture(setup_cases, tmp_path, monkeypatch)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["metadata"].pop(field)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    manifest["entries"][0]["result_sha256"] = file_hash(result_path)
    prior.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="resume provider/base URL changed or missing"):
        run_batch(dataset, tmp_path / "resumed.csv", settings, max_total_calls=1, resume_manifest=prior)


@pytest.mark.parametrize("field,value", [
    ("provider", "tokenrouter"),
    ("base_url", "https://different-attempt.example/v1"),
    ("base_url", None),
])
def test_resume_requires_attempt_route_to_agree_with_result_metadata(setup_cases, tmp_path, monkeypatch, field, value):
    dataset, settings, prior, _, result_path = _resume_routing_fixture(setup_cases, tmp_path, monkeypatch)
    attempt_path = result_path.parent / "attempt_01.json"
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    if value is None:
        attempt["request"].pop(field)
    else:
        attempt["request"][field] = value
    attempt_path.write_text(json.dumps(attempt), encoding="utf-8")
    with pytest.raises(ValueError, match="resume attempt provider/base URL changed or missing"):
        run_batch(dataset, tmp_path / "resumed.csv", settings, max_total_calls=1, resume_manifest=prior)


def test_preflight_reports_missing_facts_without_calls(setup_cases):
    dataset, raw, settings = setup_cases
    raw["products"] = []
    dataset.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    result = preflight(dataset, settings)
    assert result["provider_calls"] == 0
    assert result["ready_count"] == 0
    assert result["entries"][0]["error"]["code"] == "MISSING_FACTS"


def test_verifier_accepts_mock_provenance_but_rejects_as_live(setup_cases):
    _, raw, settings = setup_cases
    result = generate(raw, settings, provider=LabeledDouble())
    assert result["status"] == "success", result
    run_dir = Path(result["metadata"]["run_dir"])
    assert verify_run(run_dir) == []
    assert any("expected actual live" in value for value in verify_run(run_dir, expect_live=True))


def test_verifier_catches_tampering_and_rewritten_result_disconnected_from_raw(setup_cases):
    _, raw, settings = setup_cases
    result = generate(raw, settings, provider=LabeledDouble())
    run_dir = Path(result["metadata"]["run_dir"])
    path = run_dir / "result.json"
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["candidates"][0]["text"] = "这是一条与原始返回不相同的手工改写消息。"
    path.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
    assert any("hash mismatch" in value for value in verify_run(run_dir))
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["result.json"] = file_hash(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    defects = verify_run(run_dir)
    assert any("differ from validated attempt" in value for value in defects)
    assert any("reproduced from recorded provider content" in value for value in defects)


def test_mock_outputs_are_excluded_from_live_evaluation(setup_cases, tmp_path):
    dataset, _, settings = setup_cases
    def fake_generate(raw, bounded):
        return generate(raw, bounded, provider=LabeledDouble())
    output = tmp_path / "out.csv"
    run_batch(dataset, output, settings, max_total_calls=3, generate_fn=fake_generate)
    summary = evaluate_manifest(tmp_path / "final_manifest.json", tmp_path / "report.md")
    assert summary["source_counts"] == {"mock": 1}
    assert summary["verified_live_successes"] == 0
    assert summary["verified_candidate_count"] == 0
    assert not any(summary["case_errors"].values())


def test_reviews_require_scores_evidence_and_separate_ai_from_human(tmp_path):
    path = tmp_path / "reviews.csv"
    headers = ["run_id", "candidate_id", "reviewer_type", "reviewer", "scenario_fit", "naturalness",
               "brand_voice", "personalization_restraint", "hard_error", "evidence", "case_diversity"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerow(dict(zip(headers, ["unit-test", "option_1", "ai", "test-fixture", 4, 4, 4, 4,
                                          "false", "Test fixture evidence, not a production rating", 3])))
    ratings, errors = _review_rows(path, {("unit-test", "option_1"): {}})
    assert errors == []
    assert ratings[("unit-test", "option_1", "ai")]["usable"]
    assert ("unit-test", "option_1", "human") not in ratings
    text = path.read_text(encoding="utf-8").replace("false", "")
    path.write_text(text, encoding="utf-8")
    ratings, errors = _review_rows(path, {("unit-test", "option_1"): {}})
    assert ratings == {}
    assert any("hard_error" in value for value in errors)


def test_manifest_export_tampering_is_reported(setup_cases, tmp_path):
    dataset, _, settings = setup_cases
    output = tmp_path / "out.csv"
    manifest = run_batch(dataset, output, settings, prepare_only=True)
    csv_path = Path(manifest["csv_path"])
    csv_path.write_text("tampered", encoding="utf-8")
    summary = evaluate_manifest(tmp_path / "final_manifest.json", tmp_path / "report.md")
    assert any("export hash mismatch" in value for value in summary["integrity_errors"])


@pytest.mark.parametrize("entry", [None, "bad", {}, {"case_id": []}])
def test_corrupt_manifest_entries_fail_explicitly_without_attribute_errors(tmp_path, entry):
    path = tmp_path / "corrupt.json"
    path.write_text(json.dumps({"entries": [entry]}), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest entry"):
        evaluate_manifest(path, tmp_path / "report.md")


def test_jsonl_rejects_nonstandard_nonfinite_constants(setup_cases):
    dataset, raw, _ = setup_cases
    raw["products"][0]["price"] = float("nan")
    dataset.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_cases(dataset)
