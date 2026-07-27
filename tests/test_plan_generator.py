"""Tests for interview plan topic selection."""

from src.generator.interview_plan_generator import (
    select_prep_topics,
    STRONG_MIN,
    WEAK_MIN,
)


def _req(name: str, score: int) -> dict:
    return {
        "requirement": name,
        "match_score": score,
        "reasoning": "test",
        "gap_description": "test gap",
    }


def test_select_strong_weak_gap():
    analyses = [
        _req("Python LangChain RAG pipelines", 9),
        _req("LangSmith evaluation frameworks", 6),
        _req("Kubernetes production deployment", 3),
    ]
    strong, weak, gap = select_prep_topics(analyses)
    assert strong["match_score"] >= STRONG_MIN
    assert weak["match_score"] >= WEAK_MIN
    assert gap["match_score"] < WEAK_MIN


def test_select_handles_no_strong():
    analyses = [_req("Redis caching", 4), _req("GraphQL APIs", 6)]
    strong, weak, gap = select_prep_topics(analyses)
    assert strong is None
    assert weak is not None


def test_deterministic_plan_segment_count():
    from src.generator.interview_plan_generator import InterviewPlanGenerator

    analyses = [
        _req("Python LLM applications", 9),
        _req("LangSmith eval pipelines", 6),
        _req("Multi-agent orchestration", 3),
    ]
    strong, weak, gap = select_prep_topics(analyses)
    plan = InterviewPlanGenerator._build_deterministic_plan(
        InterviewPlanGenerator, "Engineer", "Build LLM apps", strong, weak, gap
    )
    phases = [s.phase for s in plan.segments]
    assert phases[0] == "strong"
    assert "close" in phases
    assert len(plan.segments) >= 3
