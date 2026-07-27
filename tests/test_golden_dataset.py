"""Tests for golden dataset schema (control + realism layers)."""

from pathlib import Path

from src.eval.run_golden_eval import (
    REALISM_PATH,
    load_datasets,
    load_golden_dataset,
    validate_dataset,
)

CONTROL_PATH = Path(__file__).resolve().parents[1] / "eval" / "golden" / "dataset.json"


def test_control_dataset_loads():
    data = load_golden_dataset(CONTROL_PATH)
    assert "cases" in data
    assert len(data["cases"]) == 10


def test_realism_dataset_loads():
    data = load_golden_dataset(REALISM_PATH)
    assert data.get("layer") == "realism"
    assert len(data["cases"]) == 15


def test_realism_seventy_five_labeled_requirements():
    data = load_golden_dataset(REALISM_PATH)
    total = sum(len(c["requirements"]) for c in data["cases"])
    assert total == 75


def test_both_layers_validate():
    for name, data in load_datasets("all"):
        errors = validate_dataset(data, name)
        assert errors == [], errors


def test_one_hundred_twenty_five_labeled_requirements_total():
    total = 0
    for _, data in load_datasets("all"):
        total += sum(len(c["requirements"]) for c in data["cases"])
    assert total == 125  # 50 control + 75 realism


def test_realism_cases_have_metadata():
    data = load_golden_dataset(REALISM_PATH)
    for case in data["cases"]:
        assert case.get("layer") == "realism"
        assert case.get("source"), f"{case['id']} missing source"
        assert case.get("source_category"), f"{case['id']} missing source_category"


def test_each_case_has_gap_labels():
    for _, data in load_datasets("all"):
        for case in data["cases"]:
            buckets = {r["gold_bucket"] for r in case["requirements"]}
            assert "gap" in buckets, f"{case['id']} missing gap label"


def test_control_has_duty_requirements():
    data = load_golden_dataset(CONTROL_PATH)
    cases_with_duty = [c for c in data["cases"] if c.get("duty_requirements")]
    assert len(cases_with_duty) >= 5
    total_duty = sum(len(c.get("duty_requirements", [])) for c in data["cases"])
    assert total_duty == 15


def test_score_ranges_match_buckets():
    for _, data in load_datasets("all"):
        for case in data["cases"]:
            for req in case["requirements"] + case.get("duty_requirements", []):
                lo, hi = req["gold_score_min"], req["gold_score_max"]
                b = req["gold_bucket"]
                if b == "strong":
                    assert lo >= 7, req
                elif b == "weak":
                    assert 4 <= lo <= 7, req
                elif b == "gap":
                    assert hi <= 5, req
