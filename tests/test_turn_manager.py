"""Tests for turn manager directives and session meta."""

from src.voice.turn_manager import TurnManager, TurnPhase


def _plan():
    return {
        "opening_line": "Welcome.",
        "strong_probed": "Python LLM apps",
        "weak_probed": "LangSmith eval",
        "gap_probed": "Kubernetes ops",
        "segments": [
            {"phase": "strong", "requirement": "Python LLM apps", "question": "Tell me about RAG?", "bridge_to_next": "Next weak."},
            {"phase": "weak", "requirement": "LangSmith eval", "question": "How would you eval?", "bridge_to_next": "Next gap."},
            {"phase": "gap", "requirement": "Kubernetes ops", "question": "K8s experience?", "bridge_to_next": ""},
            {"phase": "close", "requirement": "", "question": "Any questions?", "bridge_to_next": ""},
        ],
    }


def test_close_segment_no_bridge():
    tm = TurnManager(_plan())
    tm.segment_index = 3
    tm.phase = TurnPhase.AWAITING_USER
    d = tm.build_directive()
    assert "Use this bridge" not in d
    assert "verbatim" in d


def test_session_meta_tracks_phases():
    tm = TurnManager(_plan())
    tm.mark_current_segment_asked()
    tm.mark_current_segment_asked()
    meta = tm.get_session_meta()
    assert meta["strong_probed"] == "Python LLM apps"
    assert "strong" in meta["phases_reached"]
