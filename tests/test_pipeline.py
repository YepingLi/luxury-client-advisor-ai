"""Fault injection verifies behavior, not live model quality or final outputs."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from ca_tool.pipeline import exit_code, generate
from ca_tool.provider import ProviderError, ProviderReply
from ca_tool.settings import PROJECT_ROOT, Settings
from ca_tool.storage import RunStorage


def raw_request():
    return {
        "request_id": "TEST-001", "brand_id": "lanxu", "scenario": "price_objection",
        "reference_time": "2026-11-03T14:00:00+08:00",
        "client": {"id": "fictional-test-001", "preferred_name": "周女士",
                   "preferences": ["肩背"], "contact_permission": "allowed"},
        "conversation_history": [{"speaker": "client", "content": "喜欢款式，能打九折吗？"}],
        "products": [{"id": "LX-P001", "name": "云影肩包", "price": 18000, "currency": "CNY",
                      "material": "全粒面小牛皮", "color": "墨灰", "carry_options": ["肩背"]}],
        "scenario_context": {"target_product_id": "LX-P001", "objection_type": "discount_request",
                             "objection_text": "喜欢款式，能打九折吗？"},
    }


def valid_reply():
    texts = [
        "周女士，理解您对预算的考虑。云影肩包的正式价格是18000元，目前没有额外折扣，您可以慢慢考虑。",
        "谢谢您愿意分享想法，预算合适也很重要。云影肩包采用全粒面小牛皮，您想了解材质或肩背方式，我可以继续介绍。",
        "您可以先看看是否符合日常肩背的需要，不急于决定。如果方便，我们可以约时间试背，再感受一下。",
    ]
    content = {"candidates": [{"id": f"option_{i+1}", "text": text} for i, text in enumerate(texts)]}
    return ProviderReply(content=content, raw={"test_fixture": True, "content": content},
                         response_id="mock-response-1", usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                         model="mock-for-offline-tests")


class Stub:
    source = "mock"

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def complete(self, messages, schema, timeout):
        self.calls.append({"messages": messages, "timeout": timeout})
        value = self.outcomes.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


@pytest.fixture
def settings(tmp_path):
    return Settings(runs_dir=tmp_path / "runs", brand_path=PROJECT_ROOT / "config" / "brand.json")


def test_success_durable_and_same_request_never_overwrites(settings):
    from ca_tool.verify_run import verify_run
    first = generate(raw_request(), settings, provider=Stub([valid_reply()]))
    second = generate(raw_request(), settings, provider=Stub([valid_reply()]))
    assert first["status"] == second["status"] == "success"
    assert len(first["candidates"]) == 3
    assert first["run_id"] != second["run_id"]
    assert first["metadata"]["source"] == "mock"
    assert first["metadata"]["usage"]["total_tokens"] == 30
    directory = Path(first["metadata"]["run_dir"])
    assert json.loads((directory / "result.json").read_text("utf-8")) == first
    assert not verify_run(directory)
    assert verify_run(directory, expect_live=True)
    context = json.loads((directory / "context.json").read_text("utf-8"))
    assert "18000" in json.dumps(context, ensure_ascii=False)
    assert "九折" in json.dumps(context, ensure_ascii=False)


def test_output_repair_shares_budget_and_records_before_after(settings):
    bad = ProviderReply(content="broken JSON", raw={"content": "broken JSON"})
    adapter = Stub([bad, valid_reply()])
    result = generate(raw_request(), settings, provider=adapter)
    assert result["status"] == "success"
    assert result["metadata"]["attempts"] == 2
    assert len(adapter.calls) == 2
    assert "OUTPUT_SCHEMA" in json.dumps(adapter.calls[1]["messages"], ensure_ascii=False)
    assert (Path(result["metadata"]["run_dir"]) / "attempt_01.json").is_file()


@pytest.mark.parametrize("issue,expected_calls", [
    (ProviderError("AUTH_ERROR", "Authentication failed"), 1),
    (ProviderError("RATE_LIMIT", "Busy", retryable=True, retry_after=0), 3),
    (ProviderError("PROVIDER_TIMEOUT", "Timeout", retryable=True), 3),
    (ProviderReply(content="bad", raw={}), 3),
    (ProviderReply(content=None, raw={}, refusal="refused"), 1),
    (ProviderReply(content="truncated", raw={}, finish_reason="length"), 1),
])
def test_provider_failures_are_explicit_and_bounded(settings, issue, expected_calls, monkeypatch):
    monkeypatch.setattr("ca_tool.pipeline.time.sleep", lambda seconds: None)
    adapter = Stub([issue for _ in range(3)])
    result = generate(raw_request(), settings, provider=adapter)
    assert result["status"] == "error"
    assert result["candidates"] == []
    assert result["metadata"]["attempts"] == expected_calls
    assert len(adapter.calls) == expected_calls


def test_retry_after_longer_than_remaining_budget_stops_without_early_retry(settings):
    adapter = Stub([ProviderError("RATE_LIMIT", "Wait", retryable=True, retry_after=1000)])
    result = generate(raw_request(), settings, provider=adapter)
    assert result["error"]["code"] == "RATE_LIMIT"
    assert len(adapter.calls) == 1


def test_total_deadline_applies_to_retries(settings, monkeypatch):
    current = [100.0]
    monkeypatch.setattr("ca_tool.pipeline.time.monotonic", lambda: current[0])
    class SlowStub(Stub):
        def complete(self, messages, schema, timeout):
            current[0] += 0.2
            return super().complete(messages, schema, timeout)
    adapter = SlowStub([valid_reply()])
    result = generate(raw_request(), replace(settings, total_timeout_seconds=0.1), provider=adapter)
    assert result["error"]["code"] == "PROVIDER_TIMEOUT"
    assert adapter.calls[0]["timeout"] <= 0.1
    assert not result["candidates"]


def test_missing_fact_and_opt_out_make_zero_calls(settings):
    adapter = Stub([])
    raw = raw_request()
    raw["products"] = []
    missing = generate(raw, settings, provider=adapter)
    assert missing["status"] == "needs_input"
    assert missing["error"]["missing_fields"]
    raw = raw_request()
    raw["client"]["contact_permission"] = "do_not_contact"
    blocked = generate(raw, settings, provider=adapter)
    assert blocked["status"] == "blocked"
    assert missing["metadata"]["attempts"] == blocked["metadata"]["attempts"] == 0
    assert adapter.calls == []


def test_no_key_is_config_error_and_still_audited(settings):
    result = generate(raw_request(), replace(settings, api_key="", model=""))
    assert result["error"]["code"] == "CONFIG_ERROR"
    assert result["metadata"]["attempts"] == 0
    assert exit_code(result) == 2
    assert (Path(result["metadata"]["run_dir"]) / "manifest.json").is_file()


def test_usage_remains_explicitly_partial_when_next_attempt_fails(settings):
    bad = ProviderReply(content="broken JSON", raw={}, usage={"total_tokens": 3})
    adapter = Stub([bad, ProviderError("AUTH_ERROR", "injected authentication failure")])
    result = generate(raw_request(), settings, provider=adapter)
    assert result["metadata"]["attempts"] == 2
    assert result["metadata"]["usage"] == {"total_tokens": 3}
    assert "partial" in result["metadata"]["usage_unavailable_reason"]


def test_nonfinite_input_is_input_failure_with_audited_reason(settings):
    raw = raw_request()
    raw["products"][0]["price"] = float("nan")
    result = generate(raw, settings, provider=Stub([]))
    assert result["status"] == "needs_input"
    assert result["error"]["code"] == "INPUT_INVALID"
    assert result["metadata"]["attempts"] == 0


def test_cli_distinguishes_config_from_input_errors(monkeypatch, capsys):
    from ca_tool.generate import main
    monkeypatch.setenv("LLM_MAX_ATTEMPTS", "4")
    assert main(["--input", "examples/core_price_objection.json", "--preflight"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "error"
    assert result["error"]["code"] == "CONFIG_ERROR"
    assert result["metadata"]["attempts"] == 0


@pytest.mark.parametrize("failing_filename", ["input.json", "attempt_01.json", "result.json", "manifest.json"])
def test_storage_failure_never_returns_success(settings, failing_filename):
    class FailingStorage(RunStorage):
        def write_json(self, run_dir, name, value):
            if name == failing_filename:
                raise OSError("injected disk failure")
            return super().write_json(run_dir, name, value)
    result = generate(raw_request(), settings, provider=Stub([valid_reply()]),
                      storage=FailingStorage(settings.runs_dir))
    assert result["status"] == "error"
    assert result["error"]["code"] == "STORAGE_ERROR"
    assert result["candidates"] == []
    assert exit_code(result) == 4
    path = Path(result["metadata"]["run_dir"]) / "result.json"
    if path.exists():
        assert json.loads(path.read_text("utf-8"))["status"] == "error"
