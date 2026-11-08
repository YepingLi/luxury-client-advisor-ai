"""Frozen case sets must preserve coverage and their intended input contracts."""
from collections import Counter
import json
from pathlib import Path

import pytest

from ca_tool.batch import load_cases
from ca_tool.context import load_brand, prepare_context
from ca_tool.pipeline import generate
from ca_tool.schemas import Request
from ca_tool.settings import PROJECT_ROOT, Settings
from ca_tool.verify_run import canonical_hash, file_hash, verify_run

DATA = PROJECT_ROOT / "tests" / "data"


def read(name):
    return json.loads((DATA / name).read_text(encoding="utf-8-sig"))


@pytest.mark.parametrize("filename,expected_count", [("final_50.jsonl", 50), ("dev_12.jsonl", 12)])
def test_generation_sets_have_complete_facts_without_using_a_provider(filename, expected_count):
    brand = load_brand(PROJECT_ROOT / "config" / "brand.json")
    cases = load_cases(DATA / filename)
    assert len(cases) == expected_count
    for raw in cases:
        check = prepare_context(Request.model_validate(raw), brand)
        assert check["status"] == "ready", (raw["request_id"], check)
        assert check["error"] is None
        assert check["context"]["trusted_brand_policy"]["extra_discounts_allowed"] is False
    if filename == "dev_12.jsonl":
        assert set(Counter(case["scenario"] for case in cases).values()) == {2}


def test_frozen_inputs_match_manifest_hashes_and_predeclared_coverage():
    manifest = read("dataset_manifest.json")
    for filename, expected_hash in manifest["files"].items():
        assert file_hash(DATA / filename) == expected_hash, filename
    cases = load_cases(DATA / "final_50.jsonl")
    expected_hashes = {entry["case_id"]: entry["input_sha256"] for entry in manifest["final_cases"]}
    assert len(expected_hashes) == 50
    assert {case["request_id"]: canonical_hash(case) for case in cases} == expected_hashes
    expected_scenarios = {"greeting": 8, "product_recommendation": 9, "post_purchase_care": 8,
                          "reactivation": 8, "event_invitation": 8, "price_objection": 9}
    assert Counter(case["scenario"] for case in cases) == expected_scenarios == manifest["scenario_counts"]
    metadata = read("final_50.metadata.json")
    assert {entry["case_id"] for entry in metadata["cases"]} == set(expected_hashes)
    assert Counter(entry["category"] for entry in metadata["cases"]) == {
        "normal": 30, "optional_missing": 10, "difficult": 10}
    by_id = {case["request_id"]: case for case in cases}
    assert all(entry["scenario"] == by_id[entry["case_id"]]["scenario"] for entry in metadata["cases"])
    price_cases = [case for case in cases if case["scenario"] == "price_objection"]
    assert Counter(case["scenario_context"]["objection_type"] for case in price_cases) == {
        "too_expensive": 3, "channel_comparison": 3, "discount_request": 3}
    assert {case["request_id"] for case in load_cases(DATA / "dev_12.jsonl")}.isdisjoint(expected_hashes)


class MustNeverCall:
    source = "mock"
    def complete(self, *args, **kwargs):
        raise AssertionError("invalid/blocked robustness input reached the provider")


@pytest.mark.parametrize("expected", read("robustness.expected.json")["cases"], ids=lambda value: value["case_id"])
def test_robustness_inputs_match_predeclared_errors_with_zero_calls(expected, tmp_path):
    cases = {case["request_id"]: case for case in load_cases(DATA / "robustness.jsonl")}
    result = generate(cases[expected["case_id"]], Settings(runs_dir=tmp_path / "runs"), provider=MustNeverCall())
    assert result["status"] == expected["status"]
    assert result["error"]["code"] == expected["error_code"]
    assert result["metadata"]["attempts"] == expected["expected_model_calls"] == 0
    assert result["candidates"] == []
    assert verify_run(Path(result["metadata"]["run_dir"])) == []
