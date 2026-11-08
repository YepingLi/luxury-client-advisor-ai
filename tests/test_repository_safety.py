"""Index-blob and fictional-only evidence regressions; no Git mutations or APIs."""
import json
from pathlib import Path

import pytest

from scripts import check_repository_safety as safety


def fictional_record():
    return {
        "record_type": "fictional_client_advisor_test", "fictional_data": True,
        "input": {"conversation_history": [
            {"speaker": "advisor", "content": "虚构顾问介绍。"},
            {"speaker": "client", "content": "虚构客户问题。"}]},
        "candidate_replies": [{"id": f"option_{i}", "text": f"虚构测试候选{i}。"} for i in range(1, 4)],
    }


def encoded(record):
    return json.dumps(record, ensure_ascii=False).encode("utf-8")


@pytest.mark.parametrize("name", [
    "evidence/chat/visible_chat.json", "evidence/chat/screenshot.png",
    "evidence/debug/attempt_01.json", "evidence/qa_before.json",
    "evidence/baseline_runs/run/input.json", "Evidence/chat.json",
    "evidence/simulated_conversations/chat.html", "evidence/simulated_screenshots/chat.svg",
])
def test_developer_and_debug_evidence_paths_are_rejected_even_with_valid_payload(name):
    assert safety.evidence_errors(name, encoded(fictional_record()))


@pytest.mark.parametrize("name", [
    "evidence/README.md", "evidence/simulated_conversations/case.json",
    "evidence/simulated_screenshots/case.png", "evidence/simulated_screenshots/case.jpg",
    "evidence/simulated_screenshots/case.jpeg", "evidence/simulated_screenshots/case.webp",
])
def test_narrow_shared_evidence_paths_are_allowed(name):
    assert safety.evidence_errors(name, encoded(fictional_record())) == []


@pytest.mark.parametrize("mutation", [
    lambda record: record.update(record_type="developer_chat"),
    lambda record: record.update(fictional_data="true"),
    lambda record: record.update(input=[]),
    lambda record: record["input"].update(conversation_history="debug log"),
    lambda record: record["input"]["conversation_history"].append({"speaker": "developer", "content": "debug"}),
    lambda record: record["input"]["conversation_history"].append("not a turn object"),
    lambda record: record.update(candidate_replies=record["candidate_replies"][:2]),
    lambda record: record["candidate_replies"][0].update(text=" "),
])
def test_simulation_minimum_contract_rejects_mislabeled_or_incomplete_records(mutation):
    record = fictional_record()
    mutation(record)
    assert safety.evidence_errors("evidence/simulated_conversations/case.json", encoded(record))


@pytest.mark.parametrize("content", [b"not JSON", b"[]", b"\xff"])
def test_simulation_requires_a_utf8_json_object(content):
    assert safety.simulation_errors(content)


def test_checks_staged_blob_even_when_worktree_content_is_clean(tmp_path, monkeypatch):
    name = "evidence/simulated_conversations/case.json"
    worktree = tmp_path / name
    worktree.parent.mkdir(parents=True)
    worktree.write_bytes(encoded(fictional_record()))
    staged = fictional_record()
    staged["input"]["conversation_history"][0]["speaker"] = "system"
    commands = []

    def git_output(root, *arguments):
        assert root == tmp_path
        commands.append(arguments)
        if arguments[0] == "diff":
            assert "--cached" in arguments and "--diff-filter=ACMR" in arguments
            return name.encode() + b"\0"
        assert arguments == ("show", "--no-ext-diff", "--no-textconv", f":{name}")
        return encoded(staged)

    monkeypatch.setattr(safety, "git_output", git_output)
    report = safety.check_staged(tmp_path)
    assert report["staged_files_checked"] == 1
    assert any("advisor/client" in issue["issue"] for issue in report["issues"])
    assert len(commands) == 2


def test_staged_credentials_remain_rejected_and_never_printed(monkeypatch):
    secret = b"configured-test-secret"
    pattern_key = b"sk-or-v1-" + b"a" * 64
    blobs = {".env": secret, "source.py": pattern_key, "brief.pdf": b"private brief", "handoff.md": b"private plan"}

    def git_output(root, *arguments):
        if arguments[0] == "diff":
            return b"\0".join(name.encode() for name in blobs) + b"\0"
        return blobs[arguments[-1][1:]]

    monkeypatch.setattr(safety, "git_output", git_output)
    report = safety.check_staged(Path("unused"), secret)
    issues = {issue["issue"] for issue in report["issues"]}
    assert {"credential file staged", "actual configured key found", "full OpenRouter key pattern found",
            "confidential original input staged"}.issubset(issues)
    assert secret.decode() not in json.dumps(report)
    assert pattern_key.decode() not in json.dumps(report)


def test_empty_staged_change_set_does_not_read_worktree_or_deleted_blobs(tmp_path, monkeypatch):
    calls = []

    def git_output(root, *arguments):
        calls.append(arguments)
        assert arguments[0] == "diff" and "--diff-filter=ACMR" in arguments
        return b""

    monkeypatch.setattr(safety, "git_output", git_output)
    assert safety.check_staged(tmp_path) == {"staged_files_checked": 0, "issues": []}
    assert len(calls) == 1


def scan_blob(monkeypatch, name, content):
    def git_output(root, *arguments):
        if arguments[0] == "diff":
            return name.encode() + b"\0"
        assert arguments[-1] == f":{name}"
        return content

    monkeypatch.setattr(safety, "git_output", git_output)
    return safety.check_staged(Path("unused"))


def test_generic_long_sk_credential_is_rejected_without_configured_key(monkeypatch):
    token = b"sk-" + b"Ab9_xY7-" * 5
    report = scan_blob(monkeypatch, "source.txt", b"token=" + token)
    assert any("credential pattern" in issue["issue"] for issue in report["issues"])
    assert token.decode() not in json.dumps(report)


def test_google_key_is_rejected_without_configured_key_and_not_reported(monkeypatch):
    token = b"AIza" + b"A" * 35
    report = scan_blob(monkeypatch, "source.txt", token)
    assert any("Google API credential" in issue["issue"] for issue in report["issues"])
    assert token.decode() not in json.dumps(report)


def test_private_service_account_key_is_rejected(monkeypatch):
    report = scan_blob(monkeypatch, "credentials.json", b"-----BEGIN " + b"PRIVATE KEY-----")
    assert any("private credential" in issue["issue"] for issue in report["issues"])


@pytest.mark.parametrize("field", ["GOOGLE_CLOUD_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"])
def test_google_example_credentials_must_be_empty_or_placeholders(monkeypatch, field):
    assert scan_blob(monkeypatch, ".env.example", (field + "=a-private-value").encode())["issues"]
    assert not scan_blob(monkeypatch, ".env.example", (field + "=").encode())["issues"]


@pytest.mark.parametrize("field", ["OPENROUTER_API_KEY", "LLM_API_KEY", "TOKENROUTER_API_KEY"])
def test_env_example_rejects_nonplaceholder_credentials(monkeypatch, field):
    content = (field + "=not_a_clear_placeholder\n").encode()
    report = scan_blob(monkeypatch, ".env.example", content)
    assert any("example credential" in issue["issue"] for issue in report["issues"])
    assert "not_a_clear_placeholder" not in json.dumps(report)


@pytest.mark.parametrize("value", ["", "YOUR_TOKENROUTER_API_KEY", "replace_me", "example", '"YOUR_LLM_API_KEY" # local only'])
def test_env_example_allows_only_blank_or_clear_placeholders(monkeypatch, value):
    report = scan_blob(monkeypatch, ".env.example", ("TOKENROUTER_API_KEY=" + value + "\n").encode())
    assert report["issues"] == []


def test_env_example_checks_every_assignment_not_only_last_value(monkeypatch):
    content = b"LLM_API_KEY=not_a_placeholder\nLLM_API_KEY=replace_me\n"
    report = scan_blob(monkeypatch, ".env.example", content)
    assert report["issues"]


def test_env_example_rejects_short_sk_prefix_even_below_global_length_threshold(monkeypatch):
    content = b"export LLM_API_KEY='" + b"sk-" + b"short" + b"'\n"
    report = scan_blob(monkeypatch, ".env.example", content)
    assert report["issues"]


def test_guard_pattern_literals_and_constructed_fixtures_are_not_credentials(monkeypatch):
    content = b'pattern = rb"sk-[A-Za-z0-9_-]{32,}"\nfixture = b"sk-" + b"a" * 64\n'
    assert scan_blob(monkeypatch, "tests/test_credentials.py", content)["issues"] == []
