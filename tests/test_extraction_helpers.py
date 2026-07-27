"""Tests for debrief extraction helpers."""

from src.api.extraction_helpers import (
    build_study_topics,
    heuristic_debrief,
    candidate_answers,
)


def test_candidate_answers_parsing():
    t = "Interviewer: Hi\nCandidate: I built RAG\nInterviewer: Great\nCandidate: Used LangGraph"
    assert len(candidate_answers(t)) == 2


def test_study_topics_from_meta():
    meta = {"gap_probed": "LangSmith eval", "weak_probed": "Redis caching"}
    topics = build_study_topics(meta, [{"requirement": "K8s", "match_score": 2}])
    assert any("LangSmith" in t for t in topics)


def test_heuristic_debrief_has_study_topics():
    transcript = "Interviewer: Q1\nCandidate: " + "word " * 50
    meta = {
        "strong_probed": "Python",
        "phases_reached": ["strong"],
        "segments_asked": [0],
        "gap_probed": "",
        "weak_probed": "",
    }
    result = heuristic_debrief(transcript, meta, [])
    assert "overall_readiness" in result
    assert isinstance(result["study_topics"], list)
