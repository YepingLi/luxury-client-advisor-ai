"""Fixed-input batch generation, budget enforcement and submission exports."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .verify_run import canonical_hash, file_hash, verify_run

CSV_COLUMNS = [
    "Case ID", "Scenario", "Client information", "Conversation history",
    "Product information", "Output 1", "Output 2", "Output 3", "Scenario context",
    "Reference time", "Run ID", "Tool version", "Status", "Source", "Evaluation",
]


def atomic_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding=encoding, newline="")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant: {value}")


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, parse_constant=_reject_constant)
        except ValueError as exc:
            raise ValueError(f"line {line_no}: invalid JSON") from exc
        if not isinstance(value, dict) or not isinstance(value.get("request_id"), str) or not value["request_id"].strip():
            raise ValueError(f"line {line_no}: an object with a nonempty request_id is required")
        if value["request_id"] in seen:
            raise ValueError(f"duplicate Case ID: {value['request_id']}")
        seen.add(value["request_id"])
        cases.append(value)
    if not cases:
        raise ValueError("input dataset is empty")
    return cases


def preflight(input_path: Path, settings: Any) -> dict[str, Any]:
    """Validate inputs and essential facts without constructing a provider."""
    from .context import load_brand, prepare_context
    from .schemas import Request

    cases = load_cases(input_path)
    brand = load_brand(settings.brand_path)
    entries = []
    for raw in cases:
        try:
            request = Request.model_validate(raw)
            prepared = prepare_context(request, brand)
            entries.append({"case_id": raw["request_id"], "status": prepared["status"],
                            "warnings": prepared["warnings"], "error": prepared["error"]})
        except ValueError as exc:
            entries.append({"case_id": raw["request_id"], "status": "needs_input",
                            "error": {"code": "INPUT_INVALID", "message": str(exc)}})
    return {"case_count": len(cases), "provider_calls": 0,
            "ready_count": sum(e["status"] == "ready" for e in entries), "entries": entries}


def _pending(raw: dict, code: str = "LIVE_RUN_PENDING") -> dict:
    return {"request_id": raw["request_id"], "run_id": None, "scenario": raw.get("scenario"),
            "status": "pending", "candidates": [], "warnings": [],
            "error": {"code": code, "message": "No model output has been generated for this case.",
                      "missing_fields": [], "retryable": True},
            "metadata": {"source": "not_run", "attempts": 0, "input_sha256": canonical_hash(raw),
                         "quality_review_status": "pending"}}


def _result_path(result: dict, settings: Any) -> Path | None:
    run_id = result.get("run_id")
    if not run_id:
        return None
    run_dir = result.get("metadata", {}).get("run_dir")
    return (Path(run_dir) if run_dir else Path(settings.runs_dir) / run_id).resolve() / "result.json"


def run_batch(input_path: Path, output_path: Path, settings: Any, *,
              max_total_calls: int | None = None, prepare_only: bool = False,
              resume_manifest: Path | None = None,
              generate_fn: Callable[..., dict] | None = None) -> dict:
    """Execute every fixed case, preserving failures and immutable batch snapshots."""
    cases = load_cases(input_path)
    if max_total_calls is None:
        max_total_calls = getattr(settings, "max_total_calls", None)
    if not prepare_only and (max_total_calls is None or max_total_calls <= 0):
        raise ValueError("live batches require an explicit positive --max-total-calls budget")
    if prepare_only and resume_manifest:
        raise ValueError("prepare-only cannot resume generated results")
    if prepare_only:
        checked = preflight(input_path, settings)
        if checked["ready_count"] != checked["case_count"]:
            raise ValueError("prepare-only requires valid, generation-ready cases; use --preflight for details")
    resumed: dict[str, dict] = {}
    if resume_manifest:
        prior = json.loads(resume_manifest.read_text(encoding="utf-8-sig"))
        prior_entries = prior.get("entries", [])
        if len(prior_entries) != len(cases):
            raise ValueError("resume requires the identical fixed case set")
        previous = {entry["case_id"]: entry for entry in prior_entries}
        if len(previous) != len(cases):
            raise ValueError("resume manifest contains duplicate Case IDs")
        for raw in cases:
            entry = previous.get(raw["request_id"])
            if not entry or entry.get("input_sha256") != canonical_hash(raw):
                raise ValueError(f"resume input differs for {raw['request_id']}")
            if entry.get("status") == "success" and entry.get("source") == "live":
                path = Path(entry["result_path"])
                errors = verify_run(path.parent, expect_live=True)
                if errors or (entry.get("result_sha256") and file_hash(path) != entry["result_sha256"]):
                    raise ValueError(f"cannot resume damaged evidence: {raw['request_id']}")
                result = json.loads(path.read_text(encoding="utf-8-sig"))
                from .pipeline import source_hash
                if result.get("metadata", {}).get("versions", {}).get("source_sha256") != source_hash():
                    raise ValueError("resume source/configuration version changed; rerun all fixed cases without --resume-manifest")
                if result.get("metadata", {}).get("model") != settings.model:
                    raise ValueError("resume model changed; rerun all fixed cases without --resume-manifest")
                metadata = result.get("metadata", {})
                if any(metadata.get(field) != expected for field, expected in settings.route_identity.items()):
                    raise ValueError("resume provider/base URL changed or missing; rerun all fixed cases without --resume-manifest")
                attempts = result.get("metadata", {}).get("attempts", 0)
                last_attempt = json.loads((path.parent / f"attempt_{attempts:02d}.json").read_text(encoding="utf-8-sig"))
                recorded_request = last_attempt.get("request", {})
                if any(recorded_request.get(field) != expected for field, expected in settings.route_identity.items()):
                    raise ValueError("resume attempt provider/base URL changed or missing; rerun all fixed cases without --resume-manifest")
                parameters = recorded_request.get("parameters", {})
                recorded_sampling = parameters.get("generationConfig", {}) if settings.provider in {"vertex", "gemini"} else parameters
                max_tokens_field = "maxOutputTokens" if settings.provider in {"vertex", "gemini"} else "max_tokens"
                if recorded_sampling.get(max_tokens_field) != settings.max_tokens or recorded_sampling.get("temperature") != settings.temperature:
                    raise ValueError("resume sampling parameters changed; rerun all fixed cases without --resume-manifest")
                resumed[raw["request_id"]] = result
    if generate_fn is None and not prepare_only:
        from .pipeline import generate
        generate_fn = generate
    started = datetime.now(timezone.utc).isoformat()
    batch_id = "batch_" + uuid.uuid4().hex
    entries, records, rows = [], [], []
    total_calls = 0

    def checkpoint() -> None:
        """Checkpoint all fixed IDs before another potentially paid invocation."""
        pending_records = [{"input": raw, "result": _pending(raw)} for raw in cases[len(records):]]
        pending_entries = [{"case_id": raw["request_id"], "scenario": raw.get("scenario"),
                            "input_sha256": canonical_hash(raw), "run_id": None, "source": "not_run",
                            "status": "pending", "result_path": None, "result_sha256": None,
                            "tool_version": None, "versions": {}, "reused": False,
                            "error": _pending(raw)["error"]} for raw in cases[len(records):]]
        checkpoint_records = records + pending_records
        root = output_path.resolve().parent
        jsonl_path = root / f"{batch_id}_checkpoint.jsonl"
        atomic_text(jsonl_path, "".join(_json(record) + "\n" for record in checkpoint_records))
        snapshot = {"schema_version": "1.0", "batch_id": batch_id, "started_at": started,
                    "checkpoint_at": datetime.now(timezone.utc).isoformat(), "mode": "prepare_only" if prepare_only else "live",
                    "state": "in_progress", "input_file": str(input_path.resolve()),
                    "input_sha256": file_hash(input_path), "case_count": len(cases),
                    "completed_case_count": len(records), "call_budget": max_total_calls,
                    "new_provider_calls": total_calls, "results_jsonl_path": str(jsonl_path),
                    "results_jsonl_sha256": file_hash(jsonl_path), "entries": entries + pending_entries}
        atomic_text(root / f"{batch_id}_checkpoint.json", json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")

    if not prepare_only:
        checkpoint()
    for raw in cases:
        reused = raw["request_id"] in resumed
        if reused:
            result = resumed[raw["request_id"]]
        elif prepare_only:
            result = _pending(raw)
        elif total_calls >= max_total_calls:
            result = _pending(raw, "BATCH_BUDGET_EXHAUSTED")
        else:
            bounded = replace(settings, max_attempts=min(settings.max_attempts, max_total_calls - total_calls))
            result = generate_fn(raw, bounded)
            actual = result.get("metadata", {}).get("attempts", 0)
            if not isinstance(actual, int) or isinstance(actual, bool) or not 0 <= actual <= bounded.max_attempts:
                raise ValueError("pipeline returned an invalid call count; aborting batch")
            total_calls += actual
        metadata = result.get("metadata", {})
        result_path = _result_path(result, settings)
        versions = metadata.get("versions", {})
        tool_version = versions.get("tool_version", metadata.get("tool_version"))
        entry = {"case_id": raw["request_id"], "scenario": raw.get("scenario"),
                 "input_sha256": canonical_hash(raw), "run_id": result.get("run_id"),
                 "source": metadata.get("source", "not_run"), "status": result["status"],
                 "result_path": str(result_path) if result_path else None,
                 "result_sha256": file_hash(result_path) if result_path and result_path.is_file() else None,
                 "tool_version": tool_version, "versions": versions, "reused": reused,
                 "error": result.get("error")}
        entries.append(entry)
        records.append({"input": raw, "result": result})
        candidates = result.get("candidates", [])
        row = {"Case ID": raw["request_id"], "Scenario": raw.get("scenario", ""),
               "Client information": _json(raw.get("client", {})),
               "Conversation history": _json(raw.get("conversation_history", [])),
               "Product information": _json(raw.get("products", [])),
               "Scenario context": _json(raw.get("scenario_context", {})),
               "Reference time": raw.get("reference_time", ""), "Run ID": result.get("run_id"),
               "Tool version": tool_version, "Status": result["status"], "Source": entry["source"],
               "Evaluation": "pending human review"}
        for i in range(3):
            row[f"Output {i+1}"] = candidates[i]["text"] if len(candidates) > i else ""
        rows.append(row)
        if not prepare_only:
            checkpoint()
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    csv_text = buffer.getvalue()
    jsonl_text = "".join(_json(record) + "\n" for record in records)
    output_path = output_path.resolve()
    immutable_csv = output_path.with_name(f"{output_path.stem}_{batch_id}.csv")
    immutable_jsonl = output_path.with_name(f"{output_path.stem}_{batch_id}.jsonl")
    atomic_text(immutable_csv, csv_text, "utf-8-sig")
    atomic_text(immutable_jsonl, jsonl_text)
    atomic_text(output_path, csv_text, "utf-8-sig")
    atomic_text(output_path.with_suffix(".jsonl"), jsonl_text)
    manifest = {"schema_version": "1.0", "batch_id": batch_id, "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(), "mode": "prepare_only" if prepare_only else "live",
                "state": "complete",
                "input_file": str(input_path.resolve()), "input_sha256": file_hash(input_path),
                "case_count": len(cases), "call_budget": max_total_calls, "new_provider_calls": total_calls,
                "resume_manifest": str(resume_manifest.resolve()) if resume_manifest else None,
                "csv_path": str(immutable_csv), "csv_sha256": file_hash(immutable_csv),
                "results_jsonl_path": str(immutable_jsonl), "results_jsonl_sha256": file_hash(immutable_jsonl),
                "entries": entries}
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    atomic_text(output_path.parent / f"{batch_id}_manifest.json", manifest_text)
    atomic_text(output_path.parent / "final_manifest.json", manifest_text)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/test_cases.csv"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="Validate facts with zero provider calls and no output files")
    mode.add_argument("--prepare-only", action="store_true", help="Export pending cases with empty outputs; never call provider")
    parser.add_argument("--max-total-calls", type=int)
    parser.add_argument("--resume-manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        from .settings import Settings
        settings = replace(Settings.from_env(), runs_dir=args.runs_dir)
        if args.preflight:
            report = preflight(args.input, settings)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["ready_count"] == report["case_count"] else 2
        budget = args.max_total_calls if args.max_total_calls is not None else settings.max_total_calls
        if not args.prepare_only and (budget is None or budget <= 0):
            parser.error("live execution requires an explicit positive --max-total-calls; use --prepare-only for zero calls")
        print(f"cases={len(load_cases(args.input))}; maximum new provider calls={0 if args.prepare_only else budget}", file=sys.stderr)
        manifest = run_batch(args.input, args.output, settings, max_total_calls=budget,
                             prepare_only=args.prepare_only, resume_manifest=args.resume_manifest)
        print(json.dumps({"case_count": manifest["case_count"], "new_provider_calls": manifest["new_provider_calls"],
                          "mode": manifest["mode"], "manifest": str(args.output.parent / "final_manifest.json")},
                         ensure_ascii=False, indent=2))
        return 0 if args.prepare_only or all(e["status"] == "success" for e in manifest["entries"]) else 3
    except (OSError, ValueError) as exc:
        print(f"batch failed: {exc}", file=sys.stderr)
        return 4 if isinstance(exc, OSError) else 2


if __name__ == "__main__":
    raise SystemExit(main())
