"""Check index blobs for credentials and fictional-only shared evidence.

Developer conversations and debug logs do not belong in evidence. This guard
checks the narrow evidence contract; it does not prove semantic fictionality.
"""

import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ca_tool.settings import Settings


SCREENSHOT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
GENERIC_CREDENTIAL = re.compile(rb"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{32,}(?![A-Za-z0-9_-])")
GOOGLE_CREDENTIAL = re.compile(rb"(?<![A-Za-z0-9_-])AIza[A-Za-z0-9_-]{35}(?![A-Za-z0-9_-])")
EXAMPLE_KEY_ASSIGNMENT = re.compile(
    r"^\s*(?:export\s+)?(OPENROUTER_API_KEY|LLM_API_KEY|TOKENROUTER_API_KEY|GOOGLE_CLOUD_API_KEY|GOOGLE_API_KEY|GEMINI_API_KEY|GOOGLE_APPLICATION_CREDENTIALS)\s*=\s*(.*?)\s*$",
    re.IGNORECASE,
)


def git_output(root: Path, *arguments: str) -> bytes:
    return subprocess.run(["git", *arguments], cwd=root, capture_output=True, check=True).stdout


def simulation_errors(content: bytes) -> list[str]:
    """Validate the shared evidence minimum, independently of full Request."""
    try:
        record = json.loads(content.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ["simulation evidence must be UTF-8 JSON"]
    if not isinstance(record, dict):
        return ["simulation evidence must be a JSON object"]
    issues = []
    if record.get("record_type") != "fictional_client_advisor_test":
        issues.append("simulation record_type must be fictional_client_advisor_test")
    if record.get("fictional_data") is not True:
        issues.append("simulation fictional_data must be true")
    request = record.get("input")
    if not isinstance(request, dict):
        issues.append("simulation input must be an object")
    else:
        history = request.get("conversation_history")
        if not isinstance(history, list) or any(
            not isinstance(turn, dict) or turn.get("speaker") not in ("advisor", "client")
            or not isinstance(turn.get("content"), str)
            for turn in history
        ):
            issues.append("simulation history must contain only advisor/client message objects")
    candidates = record.get("candidate_replies")
    if not isinstance(candidates, list) or len(candidates) != 3 or any(
        not isinstance(candidate, dict) or not isinstance(candidate.get("text"), str)
        or not candidate["text"].strip() for candidate in candidates
    ):
        issues.append("simulation must contain exactly three candidate reply objects with text")
    return issues


def evidence_errors(name: str, content: bytes) -> list[str]:
    path = PurePosixPath(name)
    if not path.parts or path.parts[0].casefold() != "evidence":
        return []
    if name == "evidence/README.md":
        return []
    if path.parts[:2] == ("evidence", "simulated_conversations") and len(path.parts) >= 3 and path.suffix.lower() == ".json":
        return simulation_errors(content)
    if path.parts[:2] == ("evidence", "simulated_screenshots") and len(path.parts) >= 3 and path.suffix.lower() in SCREENSHOT_EXTENSIONS:
        return []
    return ["evidence permits only README, fictional conversation JSON and simulated screenshots"]


def example_credential_errors(content: bytes) -> list[str]:
    """Check every example assignment without logging or returning its value."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return ["example credential file must use UTF-8"]
    issues = []
    for line in text.splitlines():
        assignment = EXAMPLE_KEY_ASSIGNMENT.fullmatch(line)
        if assignment is None:
            continue
        field, value = assignment.groups()
        value = value.strip()
        if value.startswith(("'", '"')):
            quoted = re.fullmatch(r"(['\"])(.*?)\1\s*(?:#.*)?", value)
            if quoted:
                value = quoted.group(2)
        else:
            value = value.split("#", 1)[0].strip()
        clear_placeholder = re.fullmatch(r"YOUR_[A-Z0-9_]+", value, re.IGNORECASE)
        if not value or clear_placeholder or value.casefold() in {"replace_me", "example"}:
            continue
        issues.append(f"example credential field {field.upper()} must be blank or a clear placeholder")
    return issues


def check_staged(root: Path, key: bytes = b"") -> dict:
    # Deletions have no index blob to inspect and must remain allowed so old
    # developer/debug evidence can be removed. Renames inspect the new path.
    listed = git_output(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    names = [name.decode("utf-8") for name in listed.split(b"\0") if name]
    issues = []
    for name in names:
        path = PurePosixPath(name)
        if path.name.casefold() == ".env" or (path.name.casefold().startswith(".env.") and path.name.casefold() != ".env.example"):
            issues.append({"path": name, "issue": "credential file staged"})
        if path.suffix.lower() == ".pdf" or "handoff" in path.name.lower():
            issues.append({"path": name, "issue": "confidential original input staged"})
        # Read the staged version, never a possibly different worktree file.
        content = git_output(root, "show", "--no-ext-diff", "--no-textconv", f":{name}")
        issues.extend({"path": name, "issue": issue} for issue in evidence_errors(name, content))
        if path.name.casefold() == ".env.example":
            issues.extend({"path": name, "issue": issue} for issue in example_credential_errors(content))
        if key and key in content:
            issues.append({"path": name, "issue": "actual configured key found"})
        if re.search(rb"sk-or-v1-[a-fA-F0-9]{64}", content):
            issues.append({"path": name, "issue": "full OpenRouter key pattern found"})
        if GOOGLE_CREDENTIAL.search(content):
            issues.append({"path": name, "issue": "Google API credential pattern found"})
        if re.search(rb"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----", content):
            issues.append({"path": name, "issue": "private credential key found"})
        elif GENERIC_CREDENTIAL.search(content):
            issues.append({"path": name, "issue": "full sk- credential pattern found"})
    return {"staged_files_checked": len(names), "issues": issues}


def main():
    key = Settings.from_env().api_key.encode("utf-8")
    report = check_staged(ROOT, key)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["issues"] else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
