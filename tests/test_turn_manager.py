"""Tests for TurnManager scripted interview lines."""

from src.voice.turn_manager import TurnManager


def _plan():
    return {
        "job_title": "Quality Engineer",
        "opening_line": "Hi Alex, welcome.",
        "segments": [
            {
                "phase": "strong",
                "question": "Walk me through a Python testing project.",
                "bridge_to_next": "Let's shift to quality strategy.",
            },
            {
                "phase": "weak",
                "question": "How do you apply QA principles in test strategy?",
                "bridge_to_next": "This role also needs Playwright.",
            },
            {
                "phase": "gap",
                "question": "How would you approach learning Playwright?",
                "bridge_to_next": "",
            },
            {
                "phase": "close",
                "question": "Any questions about the role before we wrap up?",
                "bridge_to_next": "",
            },
        ],
    }


def test_build_first_ask_line_includes_ack_bridge_and_question():
    turns = TurnManager(_plan())
    turns.segment_index = 1
    line = turns.build_first_ask_line()
    assert "solid foundation" in line
    assert "quality strategy" in line
    assert "QA principles" in line


def test_question_for_repeat_is_current_segment_only():
    turns = TurnManager(_plan())
    turns.segment_index = 2
    assert "Playwright" in turns.question_for_repeat()
    assert "solid foundation" not in turns.question_for_repeat()
