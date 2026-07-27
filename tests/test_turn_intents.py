"""Tests for meta-intent detection."""

from src.voice.turn_intents import detect_meta_intent, is_filler_only


def test_repeat_at_start():
    assert detect_meta_intent("Can you repeat the question?") == "repeat"


def test_repeat_not_mid_long_answer():
    text = (
        "That's a very nice question again and I think the approach we took "
        "was to use LangGraph with multiple agents for our RAG pipeline implementation"
    )
    assert detect_meta_intent(text) is None


def test_listening_short():
    assert detect_meta_intent("You there?") == "listening"


def test_end_interview():
    assert detect_meta_intent("Let's wrap up, I'm done") == "end"


def test_filler_only():
    assert is_filler_only("Okay") is True
    assert is_filler_only("Hello") is False
    assert is_filler_only("I built a RAG system") is False


def test_hello_is_listening_not_filler():
    assert detect_meta_intent("Hello") == "listening"
    assert is_filler_only("Hello") is False


def test_repeated_hello_is_listening():
    from src.voice.turn_intents import detect_meta_intent

    assert detect_meta_intent("Hello Hello Hello") == "listening"
