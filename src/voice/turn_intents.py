"""Detect candidate meta-requests from transcribed speech."""

import re
from typing import Optional

REPEAT_PATTERNS = [
    r"^repeat (your |the )?question",
    r"^say that again",
    r"^(can|could) you repeat",
    r"^what was the question",
    r"^what did you ask",
    r"^i didn'?t catch",
    r"^didn'?t catch",
    r"^did not catch",
    r"^pardon\??",
    r"^sorry\?? what",
]

LISTENING_PATTERNS = [
    r"^are you listening",
    r"^can you hear me",
    r"^hello\??$",
    r"^is anyone there",
    r"^you there\??",
]

SKIP_NEXT_PATTERNS = [
    r"^(move|go) to (the )?next question",
    r"^next question",
    r"^skip (this|that|it)",
    r"^move on",
]

END_INTERVIEW_PATTERNS = [
    r"end the interview",
    r"stop the interview",
    r"let'?s (wrap|stop|end)",
    r"^i'?m done",
]

FILLER_PATTERN = re.compile(
    r"^(yeah|yes|yep|ok|okay|um+|uh+|hmm+|right|sure)[\s?.!,]*$",
    re.IGNORECASE,
)

META_MAX_WORDS = 12


def _matches_start(patterns, text: str) -> bool:
    lowered = text.lower().strip()
    return any(re.search(p, lowered) for p in patterns)


def is_repeat_request(text: str) -> bool:
    return _matches_start(REPEAT_PATTERNS, text) or (
        len(text.split()) <= META_MAX_WORDS and any(
            re.search(p, text.lower()) for p in REPEAT_PATTERNS
        )
    )


def is_listening_check(text: str) -> bool:
    return _matches_start(LISTENING_PATTERNS, text) or (
        len(text.split()) <= META_MAX_WORDS
        and any(re.search(p, text.lower()) for p in LISTENING_PATTERNS)
    )


def is_skip_next_request(text: str) -> bool:
    lowered = text.lower().strip()
    return any(re.search(p, lowered) for p in SKIP_NEXT_PATTERNS)


def is_end_interview_request(text: str) -> bool:
    lowered = text.lower().strip()
    return any(re.search(p, lowered) for p in END_INTERVIEW_PATTERNS)


def is_filler_only(text: str) -> bool:
    return bool(FILLER_PATTERN.match(text.strip()))


def detect_meta_intent(text: str) -> Optional[str]:
    """
    Return meta intent key or None for normal speech.

    Repeat/listening only trigger on short utterances or phrase-at-start
    to avoid false positives mid-answer.
    """
    if is_end_interview_request(text):
        return "end"
    if len(text.split()) <= META_MAX_WORDS:
        if is_listening_check(text):
            return "listening"
        if is_skip_next_request(text):
            return "skip"
        if is_repeat_request(text):
            return "repeat"
    else:
        if _matches_start(LISTENING_PATTERNS, text):
            return "listening"
        if _matches_start(REPEAT_PATTERNS, text):
            return "repeat"
    return None
