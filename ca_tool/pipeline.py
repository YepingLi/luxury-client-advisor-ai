"""The single generation path used by both CLIs; injected doubles are test-only."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from pydantic import ValidationError

from . import __version__
from .context import load_brand, prepare_context
from .prompts import OUTPUT_SCHEMA, PROMPT_VERSION, build_messages
from .provider import ProviderError, create_provider
from .schemas import Request, SCHEMA_VERSION
from .settings import PROJECT_ROOT, Settings
from .storage import RunStorage, canonical_hash, new_run_id, utc_now
from .validators import validate_output


def source_hash() -> str:
    digest = hashlib.sha256()
    paths = sorted((PROJECT_ROOT / "ca_tool").glob("*.py"))
    paths += sorted((PROJECT_ROOT / "config").glob("*.json"))
    paths += [PROJECT_ROOT / "requirements.txt"]
    for path in paths:
        if path.is_file():
            digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def error_object(code: str, message: str, *, missing_fields=None, retryable=False) -> dict:
    return {"code": code, "message": message, "missing_fields": missing_fields or [], "retryable": retryable}


def exit_code(result: dict) -> int:
    if result["status"] == "success":
        return 0
    code = (result.get("error") or {}).get("code")
    if code == "STORAGE_ERROR":
        return 4
    if result["status"] in ("blocked", "needs_input") or code in ("CONFIG_ERROR", "AUTH_ERROR"):
        return 2
    return 3


def generate(raw: dict, settings: Settings | None = None, *, provider=None, storage=None) -> dict:
    started = time.monotonic()
    run_id = new_run_id()
    source = getattr(provider, "source", "mock") if provider is not None else "live"
    versions = {"tool_version": __version__, "schema_version": SCHEMA_VERSION,
                "prompt_version": PROMPT_VERSION, "source_sha256": source_hash(),
                "evaluation_version": "eval-1.0.0"}
    metadata = {"source": source, "attempts": 0, "elapsed_seconds": 0.0,
                "versions": versions, "usage": None, "usage_unavailable_reason": "No model call made",
                "response_ids": [], "quality_review": "pending", "run_dir": None,
                "input_sha256": None, "recorded_at": utc_now()}
    result = {"status": "error", "request_id": raw.get("request_id") if isinstance(raw, dict) else None,
              "run_id": run_id, "scenario": raw.get("scenario") if isinstance(raw, dict) else None,
              "candidates": [], "warnings": [], "error": None, "metadata": metadata}
    configuration_error = None
    try:
        settings = settings or Settings.from_env()
    except (ValueError, OSError) as exc:
        configuration_error = error_object("CONFIG_ERROR", str(exc))
        # A malformed environment must not prevent an auditable failure when
        # the default local run directory is writable. Never construct a model
        # provider using these fallback settings.
        settings = Settings()
    metadata["model"] = settings.model or None
    metadata.update(settings.route_identity)
    store = storage if storage is not None else RunStorage(settings.runs_dir)
    run_dir = None
    try:
        run_dir = store.create_run(run_id)
        metadata["run_dir"] = str(run_dir)
        input_error = None
        try:
            serialized = json.dumps(raw, ensure_ascii=False, allow_nan=False)
            metadata["input_sha256"] = canonical_hash(raw)
            snapshot = {"raw": raw, "normalized": None}
        except (TypeError, ValueError):
            serialized = ""
            snapshot = {"raw": None, "normalized": None,
                        "omitted_reason": "Input was not a finite JSON value"}
            input_error = error_object("INPUT_INVALID", "Input must be a finite JSON value")
        store.write_json(run_dir, "input.json", snapshot)
        store.write_json(run_dir, "context.json", {"facts": None, "messages": [], "versions": versions})

        def finish(status: str, error=None) -> dict:
            result["status"] = status
            result["error"] = error
            if status != "success":
                result["candidates"] = []
            metadata["elapsed_seconds"] = round(time.monotonic() - started, 6)
            store.finalize(run_dir, result, versions)
            return result

        if configuration_error:
            return finish("error", configuration_error)
        if input_error:
            return finish("needs_input", input_error)
        if len(serialized.encode("utf-8")) > settings.max_input_bytes:
            return finish("needs_input", error_object("INPUT_INVALID", "Input exceeds MAX_INPUT_BYTES"))
        try:
            request = Request.model_validate(raw)
        except ValidationError as exc:
            details = [".".join(map(str, item["loc"])) + ": " + item["msg"]
                       for item in exc.errors(include_input=False, include_url=False)]
            return finish("needs_input", error_object("INPUT_INVALID", "; ".join(details)))
        normalized = request.model_dump(mode="json")
        store.write_json(run_dir, "input.json", {"raw": raw, "normalized": normalized})
        metadata["reference_time"] = normalized["reference_time"]
        metadata["products_sha256"] = canonical_hash(normalized["products"])
        try:
            brand = load_brand(settings.brand_path)
        except (ValueError, OSError) as exc:
            return finish("error", error_object("CONFIG_ERROR", f"Cannot load brand configuration: {type(exc).__name__}"))
        versions["brand_rules_version"] = brand.brand_rules_version
        metadata["brand_sha256"] = canonical_hash(brand.model_dump(mode="json"))
        check = prepare_context(request, brand)
        result["warnings"].extend(check["warnings"])
        if check["error"] is not None:
            store.write_json(run_dir, "context.json", {
                "facts": check["context"], "messages": [], "versions": versions})
            return finish(check["status"], check["error"])
        messages = build_messages(check["context"])
        store.write_json(run_dir, "context.json", {
            "facts": check["context"], "messages": messages, "versions": versions})
        try:
            adapter = provider if provider is not None else create_provider(settings)
        except ProviderError as exc:
            return finish("error", error_object(exc.code, exc.message, retryable=exc.retryable))
        repair_errors = None
        usages = []
        last_error = error_object("PROVIDER_TIMEOUT", "Total request budget exhausted", retryable=True)
        for number in range(1, settings.max_attempts + 1):
            remaining = settings.total_timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                break
            timeout = min(settings.timeout_seconds, remaining)
            current_messages = build_messages(check["context"], repair_errors)
            params = adapter.request_parameters(OUTPUT_SCHEMA) if hasattr(adapter, "request_parameters") else {"test_double": True}
            attempt = {"attempt": number, "source": source, "started_at": utc_now(),
                       "request": {**settings.route_identity,
                                   "messages": current_messages, "response_schema": OUTPUT_SCHEMA,
                                   "parameters": params, "timeout_seconds": timeout},
                       "response": None, "error": None, "validation": None}
            metadata["attempts"] = number
            delay = 0.0
            terminal = False
            valid = False
            try:
                reply = adapter.complete(current_messages, OUTPUT_SCHEMA, timeout)
                attempt["response"] = asdict(reply)
                if reply.response_id:
                    metadata["response_ids"].append(reply.response_id)
                usages.append(reply.usage)
                known = [u for u in usages if u is not None]
                if known:
                    metadata["usage"] = {
                        key: sum(u[key] for u in known if isinstance(u.get(key), (int, float)))
                        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                        if any(isinstance(u.get(key), (int, float)) for u in known)}
                    metadata["usage_unavailable_reason"] = None if len(known) == len(usages) else "Usage unavailable for some attempts; totals are partial"
                else:
                    metadata["usage_unavailable_reason"] = "Provider did not report token usage"
                metadata["response_id_unavailable_reason"] = None if reply.response_id else "Provider did not report a response ID"
                if reply.refusal or reply.finish_reason == "content_filter":
                    last_error = error_object("MODEL_REFUSAL", "Provider refused this request")
                    terminal = True
                elif reply.finish_reason == "length":
                    last_error = error_object("OUTPUT_TRUNCATED", "Provider stopped at its output token limit")
                    terminal = True
                elif time.monotonic() - started > settings.total_timeout_seconds:
                    last_error = error_object("PROVIDER_TIMEOUT", "Total request deadline exceeded", retryable=True)
                    terminal = True
                else:
                    checked = validate_output(reply.content, request, brand)
                    attempt["validation"] = checked
                    if checked["errors"]:
                        repair_errors = checked["errors"]
                        last_error = error_object("OUTPUT_INVALID", "; ".join(checked["errors"]))
                    else:
                        valid = True
                        result["candidates"] = checked["candidates"]
                        result["warnings"] = list(dict.fromkeys(result["warnings"] + checked["warnings"]))
            except ProviderError as exc:
                usages.append(None)
                metadata["usage_unavailable_reason"] = (
                    "Usage unavailable for some attempts; totals are partial"
                    if metadata["usage"] is not None else "Provider did not report token usage")
                attempt["error"] = exc.as_dict()
                last_error = error_object(exc.code, exc.message, retryable=exc.retryable)
                terminal = not exc.retryable
                delay = exc.retry_after if exc.retry_after is not None else min(2.0, 0.25 * 2 ** (number - 1))
            except Exception as exc:
                # SDK contract drift is a failure, never a template success.
                attempt["error"] = {"code": "PROVIDER_ERROR", "message": f"Unexpected adapter error: {type(exc).__name__}"}
                last_error = error_object("PROVIDER_ERROR", attempt["error"]["message"])
                terminal = True
            attempt["ended_at"] = utc_now()
            store.write_json(run_dir, f"attempt_{number:02d}.json", attempt)
            if valid:
                return finish("success")
            if terminal or number == settings.max_attempts:
                break
            remaining = settings.total_timeout_seconds - (time.monotonic() - started)
            # Do not retry earlier than Retry-After; defer long waits to caller.
            if delay >= remaining or delay > 60:
                break
            if delay > 0:
                time.sleep(delay)
        return finish("error", last_error)
    except (OSError, TypeError, ValueError) as exc:
        result["status"] = "error"
        result["candidates"] = []
        result["error"] = error_object("STORAGE_ERROR", f"Run could not be persisted: {type(exc).__name__}")
        metadata["elapsed_seconds"] = round(time.monotonic() - started, 6)
        # Partial files remain for diagnosis. A failed manifest write must not
        # leave a result claiming successful durable completion.
        if run_dir is not None:
            try:
                store.write_json(run_dir, "result.json", result)
            except (OSError, TypeError, ValueError):
                pass
        return result
