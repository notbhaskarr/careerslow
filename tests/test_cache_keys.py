"""Tests for cache key helpers."""

from src.utils.cache_keys import (
    make_pair_id,
    make_resume_id,
    make_jd_id,
    plan_cache_key,
    PLAN_VERSION,
)


def test_resume_id_stable():
    text = "Jane Doe\nPython developer"
    assert make_resume_id(text) == make_resume_id(text)
    assert make_resume_id(text).startswith("res_")


def test_jd_id_normalizes_whitespace():
    a = "Senior  Engineer\nPython"
    b = "senior engineer python"
    assert make_jd_id(a) == make_jd_id(b)


def test_pair_id_differs_by_jd():
    resume = "Same resume content"
    pair1 = make_pair_id(resume, "Job A Python")
    pair2 = make_pair_id(resume, "Job B Java")
    assert pair1 != pair2
    assert pair1.startswith("pair_")


def test_plan_cache_key_includes_version():
    pid = "pair_abc_def"
    assert PLAN_VERSION in plan_cache_key(pid)
