"""Post-interview debrief helpers (non-voice, candidate-facing)."""

import json
import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

MAX_TRANSCRIPT_CHARS = 14_000


def truncate_transcript(transcript: str, max_chars: int = MAX_TRANSCRIPT_CHARS) -> str:
    """Keep the tail of a long transcript within token limits."""
    if len(transcript) <= max_chars:
        return transcript
    return transcript[-max_chars:]


def candidate_answers(transcript: str) -> List[str]:
    """Extract non-empty Candidate lines from a saved transcript."""
    answers: List[str] = []
    for line in transcript.splitlines():
        if line.startswith("Candidate:"):
            text = line[len("Candidate:") :].strip()
            if text:
                answers.append(text)
    return answers


def _trim_words(text: str, max_words: int = 50) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"


def _answer_for_phase(answers: List[str], phases: List[str], phase: str) -> str:
    """
    Map candidate answers to a plan phase using segments_asked order.

    Args:
        answers: Ordered candidate utterances from transcript.
        phases: Phase label per segment reached (from session_meta).
        phase: Target phase name (strong, weak, gap, close).

    Returns:
        Trimmed merged answer text for that phase, or empty string.
    """
    if not answers or not phases:
        return ""
    idx = 0
    for seg_phase in phases:
        if idx >= len(answers):
            break
        if seg_phase == phase:
            parts = [answers[idx]]
            idx += 1
            while idx < len(answers) and idx < len(phases) and phases[idx] == phase:
                parts.append(answers[idx])
                idx += 1
            return _trim_words(" ".join(parts))
        idx += 1
    return ""


def _assess_topic(topic: str, answer: str) -> str:
    if not topic:
        return ""
    if not answer:
        return "This topic was reached but no clear response was captured."
    wc = len(answer.split())
    depth = "detailed" if wc >= 40 else "brief" if wc >= 15 else "very brief"
    return f"You gave a {depth} answer: {_trim_words(answer, 40)}"


def build_study_topics(session_meta: dict, gap_analyses: List[dict]) -> List[str]:
    """
    Build prioritized study topics from debrief gaps and low-scoring requirements.

    Args:
        session_meta: Voice session metadata (probed topics, phases).
        gap_analyses: Full gap analysis list from pair analysis cache.

    Returns:
        Ordered list of human-readable study topic strings.
    """
    topics: List[str] = []
    seen = set()

    def add(topic: str):
        key = topic.lower()[:80]
        if topic and key not in seen:
            seen.add(key)
            topics.append(topic)

    gap = session_meta.get("gap_probed", "")
    weak = session_meta.get("weak_probed", "")
    if gap:
        add(f"Study: {gap}")
    if weak:
        add(f"Strengthen your story for: {weak}")

    for g in sorted(gap_analyses, key=lambda x: x.get("match_score", 10)):
        score = g.get("match_score", 10)
        req = g.get("requirement", "")
        if score < 5 and req:
            add(f"Core gap to address: {req}")
        elif score < 8 and req and len(topics) < 6:
            add(f"Review: {req}")

    return topics[:8]


def heuristic_debrief(transcript: str, session_meta: dict, gap_analyses: List[dict]) -> dict:
    """
    Rule-based candidate debrief when LLM structured extraction fails.

    Returns:
        Dict matching DebriefResult field names.
    """
    phases = session_meta.get("phases_reached", [])
    answers = candidate_answers(transcript)

    strong_text = _answer_for_phase(answers, phases, "strong")
    weak_text = _answer_for_phase(answers, phases, "weak")
    gap_text = _answer_for_phase(answers, phases, "gap")

    segments_n = len(session_meta.get("segments_asked", []))
    if not answers:
        overall = "The session ended before meaningful responses were captured."
    else:
        overall = (
            f"You completed {segments_n} interview segment(s) across {len(answers)} response(s). "
            "Review the topic assessments below and use the study list to prepare further."
        )

    strong_label = session_meta.get("strong_probed") or session_meta.get("strength_probed", "")

    return {
        "strong_topic": strong_label,
        "strong_assessment": _assess_topic(strong_label, strong_text),
        "weak_topic": session_meta.get("weak_probed", ""),
        "weak_assessment": _assess_topic(session_meta.get("weak_probed", ""), weak_text),
        "gap_topic": session_meta.get("gap_probed", ""),
        "gap_assessment": _assess_topic(session_meta.get("gap_probed", ""), gap_text),
        "communication_notes": _trim_words(answers[-1], 30) if answers else "",
        "overall_readiness": overall,
        "study_topics": build_study_topics(session_meta, gap_analyses),
    }


def parse_extraction_json(raw: str) -> Optional[dict]:
    """Parse a JSON object from raw LLM text (with optional markdown fences)."""
    if not raw:
        return None
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            logger.debug("Failed to parse debrief JSON from LLM response")
    return None
