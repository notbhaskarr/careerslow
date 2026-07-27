"""Explicit turn state machine — single owner of interview flow."""

from enum import Enum
from typing import Optional


class TurnPhase(str, Enum):
    OPENING = "opening"
    AI_SPEAKING = "ai_speaking"
    AWAITING_USER = "awaiting_user"
    USER_SPEAKING = "user_speaking"
    CLOSED = "closed"


class TurnManager:
    """
    Drives segment progression from the pre-built InterviewPlan.

    One substantive user answer while awaiting advances exactly one segment.
    """

    def __init__(self, plan: dict):
        self.plan = plan
        self.segments = plan.get("segments", [])
        self.segment_index = 0
        self.phase = TurnPhase.OPENING
        self.segments_asked: set[int] = set()

    @property
    def strong_probed(self) -> str:
        return self.plan.get("strong_probed") or self.plan.get("strength_probed", "")

    @property
    def weak_probed(self) -> str:
        return self.plan.get("weak_probed", "")

    @property
    def gap_probed(self) -> str:
        return self.plan.get("gap_probed", "")

    def mark_current_segment_asked(self):
        """Record that the interviewer spoke the current planned segment."""
        if self.segment_index < len(self.segments):
            self.segments_asked.add(self.segment_index)

    def get_probed_topics(self) -> tuple[str, str, str]:
        """
        Resolve strong/weak/gap requirement strings from segments actually asked.

        Returns:
            (strong_probed, weak_probed, gap_probed) — empty if segment not reached.
        """
        strong = weak = gap = ""
        for idx in sorted(self.segments_asked):
            if idx >= len(self.segments):
                continue
            seg = self.segments[idx]
            phase = self._norm_phase(seg.get("phase", ""))
            req = seg.get("requirement", "")
            if phase == "strong" and not strong:
                strong = req or self.strong_probed
            elif phase == "weak" and not weak:
                weak = req or self.weak_probed
            elif phase == "gap" and not gap:
                gap = req or self.gap_probed
        return strong, weak, gap

    def get_session_meta(self) -> dict:
        """Export segment tracking for Redis and post-interview debrief."""
        strong, weak, gap = self.get_probed_topics()
        phases = []
        for idx in sorted(self.segments_asked):
            if idx < len(self.segments):
                phases.append(self._norm_phase(self.segments[idx].get("phase", "")))
        return {
            "segments_asked": sorted(self.segments_asked),
            "max_segment_reached": max(self.segments_asked) if self.segments_asked else -1,
            "strong_probed": strong,
            "weak_probed": weak,
            "gap_probed": gap,
            # Legacy keys for extraction prompt templates
            "strength_probed": strong,
            "phases_reached": phases,
        }

    @staticmethod
    def _norm_phase(phase: str) -> str:
        """Map legacy phase names to current schema."""
        if phase == "strength":
            return "strong"
        if phase == "behavioral":
            return "weak"
        return phase

    def current_segment(self) -> Optional[dict]:
        if self.segment_index < len(self.segments):
            return self.segments[self.segment_index]
        return None

    def on_ai_started(self):
        self.phase = TurnPhase.AI_SPEAKING

    def on_playback_finished(self):
        if self.segment_index >= len(self.segments):
            self.phase = TurnPhase.CLOSED
        else:
            self.phase = TurnPhase.AWAITING_USER

    def on_user_started_speaking(self):
        if self.phase in (TurnPhase.AWAITING_USER, TurnPhase.USER_SPEAKING):
            self.phase = TurnPhase.USER_SPEAKING

    def on_user_finished_speaking(self):
        if self.phase == TurnPhase.USER_SPEAKING:
            self.phase = TurnPhase.AWAITING_USER

    def advance_segment(self):
        self.segment_index += 1

    def build_directive(self, *, silence_nudge: bool = False, repeat_question: bool = False) -> str:
        """
        Build the per-turn system directive sent to the voice LLM.

        Returns:
            Directive string instructing the interviewer what to say next.
        """
        if repeat_question:
            seg = self.current_segment()
            question = seg.get("question", "") if seg else ""
            last_ai = self._last_interviewer_question()
            if last_ai:
                question = last_ai
            return (
                "[INTERVIEWER DIRECTIVE — REPEAT QUESTION]\n"
                "The candidate asked you to repeat your question.\n"
                f"Repeat exactly this question and nothing else: {question}\n"
                "Max 20 words. Do not advance to a new topic."
            )

        if silence_nudge:
            return (
                "[INTERVIEWER DIRECTIVE — SILENCE NUDGE]\n"
                "The candidate has been quiet. Say one short encouraging line like "
                "'Take your time' or gently repeat your last question in under 15 words."
            )

        if self.phase == TurnPhase.OPENING:
            seg = self.current_segment()
            opening = self.plan.get("opening_line", "Welcome to your mock interview.")
            q = seg["question"] if seg else "Tell me about your background."
            return (
                f"[INTERVIEWER DIRECTIVE — OPENING]\n"
                f"Say briefly: {opening}\n"
                f"Then ask exactly this question verbatim: {q}\n"
                f"Max 25 words total. Do not invent new topics."
            )

        seg = self.current_segment()
        if not seg:
            return (
                "[INTERVIEWER DIRECTIVE — CLOSE]\n"
                "Thank the candidate warmly and close the interview in under 25 words."
            )

        phase = self._norm_phase(seg.get("phase", ""))
        question = seg.get("question", "")

        directive = (
            f"[INTERVIEWER DIRECTIVE — segment {self.segment_index}, phase={phase}]\n"
            f"Acknowledge their answer in one short phrase.\n"
        )

        # No bridge on close; no bridge from prev when current is close
        if phase != "close":
            prev = self.segments[self.segment_index - 1] if self.segment_index > 0 else None
            bridge = prev.get("bridge_to_next", "") if prev else ""
            if bridge and self._norm_phase(prev.get("phase", "")) != "close":
                directive += f"Use this bridge: {bridge}\n"

        directive += f"Ask exactly this question verbatim: {question}\n"
        directive += "Max 25 words total. Do not invent new topics or add extra questions."
        if phase == "gap":
            directive += " Use a coaching tone — help them see what to prep."
        elif phase == "weak":
            directive += " Help them position adjacent experience honestly."
        return directive

    def should_nudge(self) -> bool:
        return self.phase == TurnPhase.AWAITING_USER

    def _last_interviewer_question(self) -> str:
        """Last planned segment question (fallback when repeating)."""
        if self.segment_index > 0:
            prev = self.segments[self.segment_index - 1]
            return prev.get("question", "")
        seg = self.current_segment()
        return seg.get("question", "") if seg else ""
