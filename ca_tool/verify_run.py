"""Check local provenance before treating a run as evaluation evidence.

Hashes detect changed files, not a malicious author who rewrites the manifest too.
The verifier does not certify semantic quality or that a provider was truthful.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _vertex_response_errors(response: dict, label: str = "Vertex") -> list[str]:
    """Reproduce native normalization for fixtures as well as live responses."""
    from .provider import ProviderError
    from .vertex_provider import normalize_vertex_response

    try:
        original = normalize_vertex_response(response.get("raw"))
    except (ProviderError, ValueError, TypeError, AttributeError):
        return [f"{label} success requires original raw generateContent response"]
    return [f"normalized {label} {field} differs from original raw response"
            for field in ("content", "refusal", "finish_reason", "response_id", "model", "usage")
            if response.get(field) != getattr(original, field)]


def _vertex_route_errors(metadata: dict, request: dict, attempt_name: str) -> list[str]:
    errors = []
    route_fields = ("provider", "base_url", "vertex_location", "vertex_auth_mode", "vertex_project_sha256")
    for field in route_fields:
        if field not in metadata or field not in request or request[field] != metadata[field]:
            errors.append(f"Vertex route {field} missing or differs between metadata and request: {attempt_name}")
    project_hash = metadata.get("vertex_project_sha256")
    if project_hash is not None and (not isinstance(project_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", project_hash)):
        errors.append("Vertex project identity must be a SHA-256 hash or null")
    private_fields = ("vertex_project", "vertex_credentials_path", "credentials_path", "GOOGLE_APPLICATION_CREDENTIALS",
                      "api_key", "authorization", "Authorization", "x-goog-api-key")
    if any(field in metadata or field in request for field in private_fields):
        errors.append("Vertex route snapshots must omit project identifiers, credential paths and credentials")
    return errors


def _gemini_route_errors(metadata: dict, request: dict, attempt_name: str) -> list[str]:
    errors = []
    for field in ("provider", "base_url"):
        if field not in metadata or field not in request or request[field] != metadata[field]:
            errors.append(f"Gemini route {field} missing or differs between metadata and request: {attempt_name}")
    if metadata.get("base_url") != "https://generativelanguage.googleapis.com/v1beta":
        errors.append("Gemini run must use the Gemini Developer API endpoint")
    private_fields = ("vertex_project", "vertex_location", "vertex_auth_mode", "vertex_project_sha256", "vertex_credentials_path",
                      "api_key", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_API_KEY", "GOOGLE_CLOUD_PROJECT",
                      "GOOGLE_CLOUD_LOCATION", "GOOGLE_APPLICATION_CREDENTIALS", "VERTEX_AUTH_MODE", "credentials_path",
                      "authorization", "Authorization", "x-goog-api-key")
    if any(field in metadata or field in request for field in private_fields):
        errors.append("Gemini snapshots must omit Cloud routing and credentials")
    return errors


def verify_run(run_dir: str | Path, expect_live: bool = False) -> list[str]:
    """Return every detected integrity/association defect; [] means checks passed."""
    root = Path(run_dir).resolve()
    errors: list[str] = []
    try:
        manifest = _read_json(root / "manifest.json")
    except (OSError, ValueError) as exc:
        return [f"manifest.json unreadable: {type(exc).__name__}"]
    if not isinstance(manifest, dict):
        return ["manifest must be an object"]
    files = manifest.get("files")
    if not isinstance(files, dict):
        return ["manifest.files must map local filenames to SHA-256 hashes"]
    for name in ("input.json", "result.json"):
        if name not in files:
            errors.append(f"required file absent from manifest: {name}")
    loaded: dict[str, Any] = {}
    for name, digest in files.items():
        if not isinstance(name, str) or not isinstance(digest, str):
            errors.append("manifest file names and hashes must be strings")
            continue
        target = (root / name).resolve()
        if not target.is_relative_to(root) or Path(name).is_absolute() or name == "manifest.json":
            errors.append(f"unsafe or self-referencing manifest path: {name}")
            continue
        try:
            if file_hash(target) != digest:
                errors.append(f"file hash mismatch: {name}")
            if name.endswith(".json"):
                loaded[name] = _read_json(target)
        except (OSError, ValueError) as exc:
            errors.append(f"file unreadable: {name} ({type(exc).__name__})")
    for candidate in root.glob("*.json"):
        if candidate.name != "manifest.json" and candidate.name not in files:
            errors.append(f"unlisted JSON file: {candidate.name}")
    result = loaded.get("result.json")
    input_record = loaded.get("input.json")
    if not isinstance(result, dict) or not isinstance(input_record, dict):
        return errors + ["input and result must be JSON objects"]
    metadata = result.get("metadata", {})
    if not isinstance(metadata, dict):
        return errors + ["result.metadata must be an object"]
    run_id = result.get("run_id")
    if not run_id or manifest.get("run_id") != run_id or root.name != run_id:
        errors.append("run ID differs between result, manifest, and directory")
    if manifest.get("status") != result.get("status"):
        errors.append("result status differs from manifest")
    source = metadata.get("source")
    if source != manifest.get("source"):
        errors.append("result source differs from manifest")
    attempts = metadata.get("attempts")
    if not isinstance(attempts, int) or isinstance(attempts, bool) or not 0 <= attempts <= 3:
        errors.append("metadata.attempts must be an integer from zero to three")
        attempts = 0
    attempt_names = sorted(name for name in loaded if re.fullmatch(r"attempt_\d{2}\.json", name))
    if attempt_names != [f"attempt_{i:02d}.json" for i in range(1, attempts + 1)]:
        errors.append("attempt files do not match recorded call count")
    if expect_live and (source != "live" or attempts == 0):
        errors.append("expected actual live invocation, not mock or zero-call preparation")
    raw = input_record.get("raw")
    normalized = input_record.get("normalized")
    if isinstance(raw, dict):
        if raw.get("request_id") != result.get("request_id"):
            errors.append("request ID differs from original input")
        if raw.get("scenario") != result.get("scenario"):
            errors.append("scenario differs from original input")
        if metadata.get("input_sha256") and metadata["input_sha256"] != canonical_hash(raw):
            errors.append("raw input hash differs from metadata")
    if normalized is not None:
        try:
            from .schemas import Request
            expected = Request.model_validate(raw).model_dump(mode="json")
            if expected != normalized:
                errors.append("normalized input differs from validation of original input")
        except Exception as exc:
            errors.append(f"normalized input cannot be revalidated: {type(exc).__name__}")
    if manifest.get("versions") and metadata.get("versions") != manifest["versions"]:
        errors.append("versions differ between manifest and result")
    context = loaded.get("context.json")
    if attempts and not isinstance(context, dict):
        errors.append("called provider without a context snapshot")
    if isinstance(context, dict) and isinstance(normalized, dict) and context.get("facts"):
        facts = context["facts"]
        if not isinstance(facts, dict):
            errors.append("context facts must be an object")
        else:
            expected_business = {key: value for key, value in normalized.items() if key != "conversation_history"}
            if facts.get("business_facts") != expected_business:
                errors.append("context business facts differ from normalized input")
            if facts.get("untrusted_conversation_history") != normalized.get("conversation_history"):
                errors.append("context history differs from normalized input")
            brand = facts.get("trusted_brand_policy")
            if metadata.get("brand_sha256") and canonical_hash(brand) != metadata["brand_sha256"]:
                errors.append("context brand differs from recorded brand hash")
            if context.get("versions") != metadata.get("versions"):
                errors.append("context versions differ from result")
    if result.get("status") == "success":
        candidates = result.get("candidates")
        if attempts == 0 or "context.json" not in loaded:
            errors.append("success requires invocation evidence and context snapshot")
        if not isinstance(candidates, list) or len(candidates) != 3:
            errors.append("success must have exactly three candidates")
        else:
            ids = [c.get("id") if isinstance(c, dict) else None for c in candidates]
            if ids != ["option_1", "option_2", "option_3"]:
                errors.append("candidate IDs must be option_1, option_2, option_3 in order")
            normalized_texts = []
            for candidate in candidates:
                text = candidate.get("text") if isinstance(candidate, dict) else None
                if not isinstance(text, str) or not text.strip() or not re.search(r"[\u3400-\u9fff]", text):
                    errors.append("candidate must be nonempty Chinese text")
                elif len(text) > 160:
                    errors.append("candidate exceeds 160 Unicode characters")
                else:
                    normalized_texts.append("".join(c for c in text if c.isalnum()))
            if len(normalized_texts) != len(set(normalized_texts)):
                errors.append("candidate text repeats after punctuation/whitespace normalization")
        if attempt_names:
            final_attempt = loaded[attempt_names[-1]]
            if isinstance(final_attempt, dict):
                validation = final_attempt.get("validation")
                response = final_attempt.get("response")
                if not isinstance(validation, dict) or validation.get("errors") != []:
                    errors.append("success has no passing validation in final attempt")
                elif validation.get("candidates") != candidates:
                    errors.append("final candidates differ from validated attempt")
                if not isinstance(response, dict) or response.get("refusal") or response.get("finish_reason") in ("length", "content_filter"):
                    errors.append("success has no complete non-refused provider response")
                elif isinstance(context, dict) and isinstance(context.get("facts"), dict):
                    try:
                        from .schemas import Brand, Request
                        from .validators import validate_output
                        checked = validate_output(response.get("content"), Request.model_validate(raw),
                                                  Brand.model_validate(context["facts"]["trusted_brand_policy"]))
                        if checked["errors"] or checked["candidates"] != candidates:
                            errors.append("final output cannot be reproduced from recorded provider content")
                    except (KeyError, ValueError, TypeError):
                        errors.append("recorded provider output cannot be independently validated")
                final_request = final_attempt.get("request", {})
                is_vertex = metadata.get("provider") == "vertex" or (
                    isinstance(final_request, dict) and final_request.get("provider") == "vertex")
                is_gemini = metadata.get("provider") == "gemini" or (
                    isinstance(final_request, dict) and final_request.get("provider") == "gemini")
                if is_gemini and isinstance(response, dict):
                    errors.extend(_vertex_response_errors(response, label="Gemini"))
                elif is_vertex and isinstance(response, dict):
                    errors.extend(_vertex_response_errors(response))
                elif source == "live" and isinstance(response, dict):
                    provider_raw = response.get("raw")
                    try:
                        choice = provider_raw["choices"][0]
                        message = choice["message"]
                        comparisons = [("content", message.get("content")), ("refusal", message.get("refusal")),
                                       ("finish_reason", choice.get("finish_reason")), ("response_id", provider_raw.get("id")),
                                       ("model", provider_raw.get("model")), ("usage", provider_raw.get("usage"))]
                        for field, original in comparisons:
                            if response.get(field) != original:
                                errors.append(f"normalized provider {field} differs from original raw response")
                    except (KeyError, IndexError, TypeError, AttributeError):
                        errors.append("live success requires original raw chat-completion response")
    elif result.get("candidates") != []:
        errors.append("non-success result must have no candidates")
    for name in attempt_names:
        attempt = loaded[name]
        if not isinstance(attempt, dict):
            errors.append(f"attempt is not an object: {name}")
        else:
            if attempt.get("source") is not None and attempt["source"] != source:
                errors.append(f"attempt source differs from result: {name}")
            expected_number = int(name[8:10])
            if attempt.get("attempt") != expected_number:
                errors.append(f"attempt number differs from filename: {name}")
            request_snapshot = attempt.get("request", {})
            if metadata.get("provider") == "gemini" or (isinstance(request_snapshot, dict) and request_snapshot.get("provider") == "gemini"):
                errors.extend(_gemini_route_errors(metadata, request_snapshot if isinstance(request_snapshot, dict) else {}, name))
            if metadata.get("provider") == "vertex" or (isinstance(request_snapshot, dict) and request_snapshot.get("provider") == "vertex"):
                errors.extend(_vertex_route_errors(metadata, request_snapshot if isinstance(request_snapshot, dict) else {}, name))
            messages = request_snapshot.get("messages", []) if isinstance(request_snapshot, dict) else []
            if not isinstance(messages, list) or len(messages) != 2 or not all(isinstance(m, dict) for m in messages) or [m.get("role") for m in messages] != ["system", "user"]:
                errors.append(f"provider messages must contain only trusted system and serialized user context: {name}")
            if isinstance(context, dict) and name == "attempt_01.json" and messages != context.get("messages"):
                errors.append("first attempt messages differ from context snapshot")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expect-live", action="store_true")
    args = parser.parse_args(argv)
    errors = verify_run(args.run_dir, args.expect_live)
    print(json.dumps({"valid": not errors, "errors": errors}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
