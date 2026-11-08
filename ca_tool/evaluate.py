"""Evaluate provenance and objective generation metrics; never invent human scores."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .batch import atomic_text, load_cases
from .verify_run import canonical_hash, file_hash, verify_run

EVALUATION_VERSION = "eval-1.0.0"
SCORE_COLUMNS = ["scenario_fit", "naturalness", "brand_voice", "personalization_restraint"]
REVIEW_COLUMNS = ["case_id", "run_id", "candidate_id", "reviewer_type", "reviewer", *SCORE_COLUMNS,
                  "hard_error", "evidence", "case_diversity"]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _percentage(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({numerator / denominator:.1%})" if denominator else "pending (no eligible outputs)"


def recorded_cost(run_dir: Path, attempts: int) -> tuple[Decimal, int]:
    """Aggregate reported charges, preserving unavailable-attempt coverage."""
    total = Decimal("0")
    known = 0
    for number in range(1, attempts + 1):
        try:
            attempt = _load_json(run_dir / f"attempt_{number:02d}.json")
            reported = attempt["response"]["raw"]["usage"]["cost"]
            if isinstance(reported, bool):
                continue
            amount = Decimal(str(reported))
            if amount.is_finite() and amount >= 0:
                total += amount
                known += 1
        except (OSError, ValueError, TypeError, KeyError, InvalidOperation):
            pass
    return total, known


def _review_rows(path: Path | None, eligible: dict[tuple[str, str], dict]) -> tuple[dict, list[str]]:
    ratings: dict[tuple[str, str, str], dict] = {}
    errors = []
    if path is None:
        return ratings, errors
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = set(REVIEW_COLUMNS) - {"case_id"}
        if not required.issubset(reader.fieldnames or []):
            return {}, ["review file is missing required columns"]
        for line_no, row in enumerate(reader, 2):
            if not any(str(row.get(c, "")).strip() for c in SCORE_COLUMNS):
                continue  # unfilled template row is explicitly pending
            try:
                key = (row["run_id"].strip(), row["candidate_id"].strip(), row["reviewer_type"].strip())
                if key[:2] not in eligible:
                    raise ValueError("rating does not refer to a verified live output")
                if key[2] not in ("human", "ai") or not row["reviewer"].strip():
                    raise ValueError("reviewer_type and named reviewer are required")
                if key in ratings:
                    raise ValueError("duplicate rating for run/candidate/reviewer_type")
                scores = [int(row[c]) for c in SCORE_COLUMNS]
                diversity = int(row["case_diversity"])
                if not all(1 <= value <= 5 for value in [*scores, diversity]):
                    raise ValueError("scores must be integers from one to five")
                hard_value = row["hard_error"].strip().lower()
                if hard_value not in ("true", "false"):
                    raise ValueError("hard_error must explicitly be true or false")
                if not row["evidence"].strip():
                    raise ValueError("review evidence is required")
                hard_error = hard_value == "true"
                ratings[key] = {"scores": scores, "hard_error": hard_error,
                                "usable": not hard_error and min(scores) >= 3 and sum(scores) >= 16,
                                "diversity": diversity, "reviewer": row["reviewer"], "evidence": row["evidence"]}
            except (TypeError, ValueError, AttributeError) as exc:
                errors.append(f"review CSV row {line_no}: {exc}")
    by_case = defaultdict(set)
    for (run_id, _, reviewer_type), rating in ratings.items():
        by_case[(run_id, reviewer_type)].add(rating["diversity"])
    for key, values in by_case.items():
        if len(values) != 1:
            errors.append(f"inconsistent case_diversity for {key[0]} ({key[1]})")
    return ratings, errors


def evaluate_manifest(manifest_path: Path, output_path: Path, reviews_path: Path | None = None) -> dict:
    if reviews_path and reviews_path.resolve() == (output_path.resolve().parent / "human_reviews_template.csv"):
        raise ValueError("copy human_reviews_template.csv to a separate review file before passing --reviews")
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("entries"), list):
        raise ValueError("manifest must contain an entries array")
    entries = manifest["entries"]
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"manifest entry {position} must be an object")
        for field in ("case_id", "scenario", "status", "source", "input_sha256"):
            if not isinstance(entry.get(field), str) or not entry[field]:
                raise ValueError(f"manifest entry {position} requires nonempty {field}")
        if entry.get("result_path") is not None and not isinstance(entry["result_path"], str):
            raise ValueError(f"manifest entry {position} result_path must be a string or null")
    integrity_errors = []
    input_cases: dict[str, dict] = {}
    try:
        input_path = Path(manifest["input_file"])
        if file_hash(input_path) != manifest["input_sha256"]:
            integrity_errors.append("fixed input file hash mismatch")
        input_cases = {case["request_id"]: case for case in load_cases(input_path)}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        integrity_errors.append(f"fixed input file cannot be checked: {type(exc).__name__}")
    seen = [entry.get("case_id") for entry in entries if isinstance(entry, dict)]
    if len(seen) != len(entries) or len(set(seen)) != len(seen):
        integrity_errors.append("manifest has invalid or duplicate Case IDs")
    if set(seen) != set(input_cases):
        integrity_errors.append("manifest case IDs differ from the fixed input set")
    if manifest.get("case_count") != len(entries):
        integrity_errors.append("manifest case_count differs from entries")
    for label, hash_key in (("csv_path", "csv_sha256"), ("results_jsonl_path", "results_jsonl_sha256")):
        try:
            if file_hash(Path(manifest[label])) != manifest[hash_key]:
                integrity_errors.append(f"export hash mismatch: {label}")
        except (OSError, KeyError, TypeError):
            integrity_errors.append(f"export unavailable: {label}")
    records: dict[str, dict] = {}
    try:
        for line in Path(manifest["results_jsonl_path"]).read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("input"), dict) or not isinstance(record.get("result"), dict) or not isinstance(record["result"].get("metadata"), dict):
                raise ValueError("full result JSONL rows require input/result objects")
            key = record["input"]["request_id"]
            if key in records:
                integrity_errors.append(f"duplicate JSONL input: {key}")
            records[key] = record
        if set(records) != set(seen):
            integrity_errors.append("JSONL case IDs differ from manifest")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        integrity_errors.append(f"full result JSONL cannot be checked: {type(exc).__name__}")
    verified_successes = []
    per_case_errors: dict[str, list[str]] = {}
    results: dict[str, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        case_id = entry.get("case_id", "<unknown>")
        errors = []
        raw = input_cases.get(case_id)
        if raw is None or canonical_hash(raw) != entry.get("input_sha256"):
            errors.append("case input hash mismatch")
        record = records.get(case_id)
        if record and canonical_hash(record.get("input")) != entry.get("input_sha256"):
            errors.append("JSONL input differs from fixed input")
        if entry.get("result_path"):
            path = Path(entry["result_path"])
            try:
                if file_hash(path) != entry.get("result_sha256"):
                    errors.append("result file hash differs from batch manifest")
                result = _load_json(path)
                if not isinstance(result, dict) or not isinstance(result.get("metadata"), dict):
                    raise ValueError("result and its metadata must be objects")
                errors.extend(verify_run(path.parent, expect_live=entry.get("status") == "success" and entry.get("source") == "live"))
                for field in ("run_id", "status"):
                    if result.get(field) != entry.get(field):
                        errors.append(f"result {field} differs from batch entry")
                if result.get("request_id") != case_id or result.get("metadata", {}).get("source") != entry.get("source"):
                    errors.append("result identity/source differs from batch entry")
                if entry.get("versions", {}) != result.get("metadata", {}).get("versions", {}):
                    errors.append("result versions differ from batch entry")
                if record and result != record.get("result"):
                    errors.append("result JSONL differs from original run")
                results[case_id] = result
            except (OSError, ValueError, TypeError, KeyError) as exc:
                errors.append(f"result unavailable: {type(exc).__name__}")
        else:
            result = record.get("result", {}) if record else {}
            results[case_id] = result
            if entry.get("status") != "pending" or entry.get("source") != "not_run":
                errors.append("executed result is missing its original run path")
            if result.get("candidates") != [] or result.get("metadata", {}).get("attempts") != 0:
                errors.append("pending case contains invented outputs or calls")
            if result.get("status") != entry.get("status") or result.get("request_id") != case_id or result.get("metadata", {}).get("source") != entry.get("source"):
                errors.append("pending JSONL result identity/source/status differs from batch entry")
        per_case_errors[case_id] = errors
        if not errors and not integrity_errors and entry.get("source") == "live" and entry.get("status") == "success":
            verified_successes.append(entry)
    eligible = {}
    for entry in verified_successes:
        for candidate in results[entry["case_id"]]["candidates"]:
            eligible[(entry["run_id"], candidate["id"])] = candidate
    ratings, review_errors = _review_rows(reviews_path, eligible)
    if review_errors:
        ratings = {}  # a malformed review file must never produce reassuring scores
    review_summary = {}
    for reviewer_type in ("human", "ai"):
        subset = {key: value for key, value in ratings.items() if key[2] == reviewer_type}
        reviewed_cases = passed_cases = 0
        for entry in verified_successes:
            values = [subset.get((entry["run_id"], f"option_{i}", reviewer_type)) for i in range(1, 4)]
            if all(values):
                reviewed_cases += 1
                passed_cases += int(sum(v["usable"] for v in values) >= 2 and values[0]["diversity"] >= 3)
        review_summary[reviewer_type] = {
            "reviewed_candidates": len(subset), "usable_candidates": sum(v["usable"] for v in subset.values()),
            "hard_errors": sum(v["hard_error"] for v in subset.values()),
            "reviewed_cases": reviewed_cases, "passed_cases": passed_cases,
            "status": "complete" if eligible and len(subset) == len(eligible) else "pending",
        }
        complete = bool(eligible) and len(subset) == len(eligible) and len(eligible) == len(entries) * 3
        review_summary[reviewer_type]["targets"] = {
            "candidate_usability_at_least_90_percent": (sum(v["usable"] for v in subset.values()) / len(eligible) >= 0.9) if complete else None,
            "case_pass_rate_at_least_90_percent": (passed_cases / len(entries) >= 0.9) if complete else None,
            "zero_hard_errors": (not any(v["hard_error"] for v in subset.values())) if complete else None,
            "scope": "human review" if reviewer_type == "human" else "AI-assisted review only; not human approval",
        }
    by_scenario = defaultdict(Counter)
    successful_ids = {entry["case_id"] for entry in verified_successes}
    for entry in entries:
        if isinstance(entry, dict):
            by_scenario[entry.get("scenario", "unknown")]["total"] += 1
            by_scenario[entry.get("scenario", "unknown")]["success"] += int(entry.get("case_id") in successful_ids)
    sources = Counter(entry.get("source", "unknown") for entry in entries if isinstance(entry, dict))
    statuses = Counter(entry.get("status", "unknown") for entry in entries if isinstance(entry, dict))
    total_calls = 0
    elapsed = 0.0
    usage_totals: Counter = Counter()
    missing_usage = 0
    one_call_successes = 0
    versions = set()
    provider_cost = Decimal("0")
    cost_known_calls = 0
    cost_expected_calls = 0
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("source") != "live":
            continue
        result = results.get(entry["case_id"], {})
        metadata = result.get("metadata", {})
        calls = metadata.get("attempts", 0)
        if isinstance(calls, int) and not isinstance(calls, bool):
            total_calls += calls
            if not integrity_errors and not per_case_errors.get(entry["case_id"]):
                cost_expected_calls += calls
                amount, known = recorded_cost(Path(entry["result_path"]).parent, calls)
                provider_cost += amount
                cost_known_calls += known
        duration = metadata.get("elapsed_seconds")
        if isinstance(duration, (int, float)) and math.isfinite(duration):
            elapsed += duration
        if entry["case_id"] in successful_ids:
            one_call_successes += int(calls == 1)
            versions.add(json.dumps(metadata.get("versions", {}), sort_keys=True))
        usage = metadata.get("usage")
        if isinstance(usage, dict):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                    usage_totals[key] += value
        elif calls:
            missing_usage += 1
    if len(versions) > 1:
        integrity_errors.append("successful final cases use different frozen versions; rerun all final cases on one version")
    summary = {"evaluation_version": EVALUATION_VERSION, "case_count": len(entries),
               "is_complete_final_50": len(entries) == 50 and len(verified_successes) == 50 and len(versions) == 1
                                       and not integrity_errors and not any(per_case_errors.values()),
               "verified_live_successes": len(verified_successes), "verified_candidate_count": len(eligible),
               "status_counts": dict(statuses), "source_counts": dict(sources),
               "single_call_successes": one_call_successes, "multiple_call_successes": len(verified_successes) - one_call_successes,
               "final_run_provider_calls": total_calls, "elapsed_seconds_sum": elapsed,
               "usage": dict(usage_totals) if usage_totals else None, "live_runs_missing_usage": missing_usage,
               "provider_reported_cost_usd": str(provider_cost) if cost_known_calls else None,
               "cost_coverage": {"calls_with_reported_cost": cost_known_calls, "referenced_verified_live_calls": cost_expected_calls,
                                 "complete": cost_expected_calls > 0 and cost_known_calls == cost_expected_calls,
                                 "basis": "Sum of recorded OpenRouter response.raw.usage.cost; USD credits; no rate-based estimate"},
               "reviews": review_summary, "integrity_errors": integrity_errors, "case_errors": per_case_errors,
               "review_errors": review_errors}
    lines = ["# Evaluation report", "", f"Evaluation version: {EVALUATION_VERSION}.", "",
             f"Fixed denominator: **{len(entries)} cases / {len(entries) * 3} expected candidate messages**.",
             f"Verified live generation success: **{_percentage(len(verified_successes), len(entries))}**.",
             f"Actual verified candidate outputs: **{len(eligible)}**.",
             f"Final 50-case generation requirement: **{'complete' if summary['is_complete_final_50'] else 'not complete'}**.", "",
             f"Statuses: `{json.dumps(dict(statuses), sort_keys=True)}`.",
             f"Sources: `{json.dumps(dict(sources), sort_keys=True)}`. Mock records are excluded from live success and quality metrics.", "",
             "## Objective checks", "",
             "Integrity checks cover fixed input hashes, immutable exports, per-run file hashes, run associations and output structure. They do not prove absence of hallucinations or appropriate tone.",
             f"Single-call successful cases: {one_call_successes}; successful cases after multiple calls: {len(verified_successes) - one_call_successes}. Multiple calls may include network retries as well as output repair.",
             f"Provider calls in final referenced live runs: {total_calls}; calls newly made in this batch: {manifest.get('new_provider_calls', 'unknown')}.",
             f"Sum of referenced live run durations: {elapsed:.3f} seconds.",
             f"Reported token usage: {json.dumps(dict(usage_totals)) if usage_totals else 'unavailable / no live calls'}; live runs with missing usage: {missing_usage}.",
             (f"Provider-reported cost: {provider_cost} USD credits from {cost_known_calls}/{cost_expected_calls} referenced verified calls; {'complete for these calls' if cost_expected_calls > 0 and cost_known_calls == cost_expected_calls else 'partial or unavailable'}. "
              "Only recorded `response.raw.usage.cost` is summed; this is not an estimate of other development calls or account funding fees. "
              "[OpenRouter usage accounting](https://openrouter.ai/docs/cookbook/administration/usage-accounting), [USD credit basis](https://openrouter.ai/support/)."), "",
             "| Scenario | Fixed cases | Verified live successes |", "| --- | ---: | ---: |"]
    for scenario in sorted(by_scenario):
        counts = by_scenario[scenario]
        lines.append(f"| {scenario} | {counts['total']} | {counts['success']} |")
    lines.extend(["", "## Quality review", "",
                  "Candidate usability requires no hard error, all four dimension scores at least 3, and a total of at least 16/20. A case passes with at least two usable candidates and diversity at least 3/5.",
                  "The dimensions are scenario fit, naturalness, brand voice, and personalization with restraint. Human and AI ratings are reported separately. Blank scores remain pending.", ""])
    for reviewer_type in ("human", "ai"):
        quality = review_summary[reviewer_type]
        lines.extend([f"### {'Human' if reviewer_type == 'human' else 'AI-assisted'} review", "",
                      f"Status: **{quality['status']}**. Reviewed messages: {quality['reviewed_candidates']}/{len(entries) * 3}; fully reviewed cases: {quality['reviewed_cases']}/{len(entries)}.",
                      f"Usability among reviewed messages: {_percentage(quality['usable_candidates'], quality['reviewed_candidates'])}.",
                      f"Pass rate among fully reviewed cases: {_percentage(quality['passed_cases'], quality['reviewed_cases'])}.",
                      f"Confirmed usable messages against the fixed output denominator: {quality['usable_candidates']}/{len(entries) * 3}; confirmed passing cases: {quality['passed_cases']}/{len(entries)}. Unreviewed outputs/cases remain pending, so rates among reviewed items alone cannot establish the final 90% targets.",
                      f"Recorded hard errors in reviewed messages: {quality['hard_errors'] if quality['reviewed_candidates'] else 'pending'}.", ""])
        lines.extend([f"Target outcomes ({'human' if reviewer_type == 'human' else 'AI-assisted only'}): `{json.dumps(quality['targets'], sort_keys=True)}`. Null means pending.", ""])
    lines.extend(["## Defects and limitations", ""])
    all_errors = [*integrity_errors, *review_errors]
    all_errors.extend(f"{case_id}: {error}" for case_id, errors in per_case_errors.items() for error in errors)
    lines.extend([f"- {error}" for error in all_errors] or ["No integrity defects detected in available records. This is not a quality approval."])
    if not verified_successes:
        lines.append("No verified live generation has been evaluated. Empty outputs are preparation records, not model results. The 150-output requirement remains pending.")
    if review_summary["human"]["status"] != "complete":
        lines.append("Human semantic review is incomplete; the 90% usability/pass targets and zero factual/policy-error target have not been established.")
    lines.extend(["Screenshots: pending; this program does not fabricate chat or evaluation screenshots.", "",
                  "The review CSV contains reviewer-supplied judgments. The software validates their format and associations, not the reviewer's identity.", ""])
    output_path = output_path.resolve()
    atomic_text(output_path, "\n".join(lines))
    atomic_text(output_path.with_suffix(".json"), json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    template = io.StringIO(newline="")
    writer = csv.DictWriter(template, fieldnames=REVIEW_COLUMNS)
    writer.writeheader()
    for entry in entries:
        for i in range(1, 4):
            writer.writerow({"case_id": entry.get("case_id"), "run_id": entry.get("run_id"),
                             "candidate_id": f"option_{i}", "reviewer_type": "human"})
    template_path = output_path.parent / "human_reviews_template.csv"
    atomic_text(template_path, template.getvalue(), "utf-8-sig")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args(argv)
    try:
        summary = evaluate_manifest(args.manifest, args.output, args.reviews)
        print(json.dumps({"case_count": summary["case_count"], "verified_live_successes": summary["verified_live_successes"],
                          "human_review": summary["reviews"]["human"]["status"],
                          "final_50_complete": summary["is_complete_final_50"]}, ensure_ascii=False, indent=2))
        return 1 if summary["integrity_errors"] or any(summary["case_errors"].values()) or summary["review_errors"] else 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
