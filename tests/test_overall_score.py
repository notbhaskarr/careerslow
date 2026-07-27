"""Unit tests for overall fit score blending."""

from src.graph.workflow import BLEND_DUTY, BLEND_NICE, BLEND_REQUIRED, _overall_fit_score
from src.schemas.document_schemas import (
    MAX_IMPORTANCE_WEIGHT,
    JobRequirement,
    ResponsibilityTheme,
    cap_importance_weight,
)


def test_full_blend_65_15_20():
    # req=8, duty=6, nice=10 → 0.65*8 + 0.15*6 + 0.20*10 = 8.1
    score = _overall_fit_score(8.0, 1.0, 6.0, 1.0, 10.0, 1.0)
    assert round(score, 2) == 8.1


def test_renormalizes_when_no_duties():
    # req + nice only → 65/85 required, 20/85 nice
    score = _overall_fit_score(8.0, 1.0, 0.0, 0.0, 10.0, 1.0)
    assert round(score, 2) == round((65 * 8 + 20 * 10) / 85, 2)


def test_required_only():
    assert _overall_fit_score(7.5, 1.0, 0.0, 0.0, 0.0, 0.0) == 7.5


def test_all_categories_absent():
    assert _overall_fit_score(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) == 0.0


def test_blend_constants_sum_to_one():
    assert round(BLEND_REQUIRED + BLEND_DUTY + BLEND_NICE, 2) == 1.0


def test_cap_importance_weight_clamps_high_values():
    assert cap_importance_weight(10) == MAX_IMPORTANCE_WEIGHT
    assert cap_importance_weight(15) == MAX_IMPORTANCE_WEIGHT


def test_cap_importance_weight_clamps_low_values():
    assert cap_importance_weight(0) == 1
    assert cap_importance_weight(-3) == 1


def test_schema_validator_caps_importance_weight():
    req = JobRequirement(skill_name="Python", importance_weight=10)
    theme = ResponsibilityTheme(theme_name="Test planning", importance_weight=10)
    assert req.importance_weight == MAX_IMPORTANCE_WEIGHT
    assert theme.importance_weight == MAX_IMPORTANCE_WEIGHT
