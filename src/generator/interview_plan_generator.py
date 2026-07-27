"""
Interview plan generator for ~5-minute candidate prep mocks.

Topic selection is deterministic (code); LLM only polishes opening and question wording.
Segments: strong → weak → gap → close (4 total).
"""

import logging
import re
from typing import List, Optional, Tuple

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.schemas.document_schemas import InterviewPlan, InterviewSegment
from src.utils.text import resume_excerpt

logger = logging.getLogger(__name__)

STRONG_MIN = 8
WEAK_MIN = 5
WEAK_MAX = 7


def _keywords(text: str) -> set:
    """Tokenize requirement text for overlap scoring."""
    tokens = re.findall(r"[a-zA-Z0-9+#.]+", text.lower())
    stop = {"and", "the", "for", "with", "using", "experience", "skills", "knowledge"}
    return {t for t in tokens if len(t) > 2 and t not in stop}


def _overlap(a: str, b: str) -> int:
    """Count shared keywords between two requirement strings."""
    return len(_keywords(a) & _keywords(b))


def select_prep_topics(
    gap_analyses: List[dict],
) -> Tuple[Optional[dict], Optional[dict], Optional[dict]]:
    """
    Pick one strong, one weak, and one gap topic for the mock interview.

    Args:
        gap_analyses: GapAnalysis dicts with match_score and requirement.

    Returns:
        (strong, weak, gap) — any may be None if no requirements in that band.
    """
    strongs = sorted(
        [g for g in gap_analyses if g.get("match_score", 0) >= STRONG_MIN],
        key=lambda g: g["match_score"],
        reverse=True,
    )
    weaks = sorted(
        [g for g in gap_analyses if WEAK_MIN <= g.get("match_score", 0) <= WEAK_MAX],
        key=lambda g: g["match_score"],
        reverse=True,
    )
    gaps = sorted(
        [g for g in gap_analyses if g.get("match_score", 0) < WEAK_MIN],
        key=lambda g: g["match_score"],
    )

    picked_strong = strongs[0] if strongs else None
    anchor = picked_strong["requirement"] if picked_strong else ""

    picked_weak = None
    if weaks:
        picked_weak = max(weaks, key=lambda g: _overlap(anchor, g["requirement"]) if anchor else 0)
    elif strongs and len(strongs) > 1:
        picked_weak = strongs[1]

    picked_gap = None
    if gaps:
        ref = picked_weak["requirement"] if picked_weak else anchor
        picked_gap = max(gaps, key=lambda g: _overlap(ref, g["requirement"]) if ref else 0)

    return picked_strong, picked_weak, picked_gap


def _format_topic_block(gap: dict) -> str:
    return (
        f"Requirement: {gap['requirement']}\n"
        f"Score: {gap['match_score']}/10\n"
        f"Evidence/Reasoning: {gap.get('reasoning', '')}\n"
        f"Gap: {gap.get('gap_description', '')}"
    )


class InterviewPlanGenerator:
    """Builds a 4-segment prep interview plan from gap analysis scores."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)
        self.structured_llm = self.llm.with_structured_output(InterviewPlan)

    async def generate(
        self,
        gap_analyses: List[dict],
        job_title: str,
        raw_jd: str,
        raw_resume: str,
        responsibilities: Optional[List[str]] = None,
    ) -> InterviewPlan:
        """
        Generate a ~5-minute mock interview plan.

        Args:
            gap_analyses: Scored requirements from LangGraph.
            job_title: Target role title.
            raw_jd: Full JD text.
            raw_resume: Full resume text (re-upload runs full pipeline with new hash).
            responsibilities: Parsed JD responsibilities (optional).

        Returns:
            InterviewPlan with exactly 4 segments when topics exist.
        """
        strong, weak, gap = select_prep_topics(gap_analyses)
        plan = self._build_deterministic_plan(job_title, raw_jd, strong, weak, gap)

        try:
            polished = await self._polish_with_llm(
                plan, job_title, raw_jd, raw_resume, responsibilities or [], strong, weak, gap
            )
            if polished and len(polished.segments) >= 3:
                return polished
        except Exception as e:
            logger.warning(f"LLM plan polish failed, using deterministic plan: {e}")

        return plan

    def _build_deterministic_plan(
        self,
        job_title: str,
        raw_jd: str,
        strong: Optional[dict],
        weak: Optional[dict],
        gap: Optional[dict],
    ) -> InterviewPlan:
        """Build segment questions and bridges without LLM."""
        s_req = strong["requirement"] if strong else "your most relevant experience"
        w_req = weak["requirement"] if weak else ""
        g_req = gap["requirement"] if gap else "a key requirement for this role"

        segments: List[InterviewSegment] = []

        segments.append(
            InterviewSegment(
                phase="strong",
                requirement=s_req,
                match_score=strong["match_score"] if strong else 0,
                resume_evidence=(strong.get("reasoning", "")[:200] if strong else ""),
                question=f"Walk me through a project where you used {s_req}.",
                bridge_to_next=(
                    f"You have adjacent experience around {w_req}."
                    if w_req
                    else f"This role also emphasizes {g_req}."
                ),
            )
        )

        if weak:
            segments.append(
                InterviewSegment(
                    phase="weak",
                    requirement=w_req,
                    match_score=weak["match_score"],
                    gap_summary=weak.get("gap_description", "")[:200],
                    question=(
                        f"Your resume touches {w_req} — how would you position that for this role?"
                    ),
                    bridge_to_next=f"Let's discuss {g_req}, which the JD calls out directly.",
                )
            )

        segments.append(
            InterviewSegment(
                phase="gap",
                requirement=g_req,
                match_score=gap["match_score"] if gap else 0,
                gap_summary=(gap.get("gap_description", "")[:200] if gap else ""),
                question=(
                    f"This role needs {g_req}. How would you approach it if hired?"
                ),
                bridge_to_next="",
            )
        )

        segments.append(
            InterviewSegment(
                phase="close",
                requirement="",
                question="Do you have any questions about the role before we wrap up?",
                bridge_to_next="",
            )
        )

        return InterviewPlan(
            opening_line=(
                f"Welcome to your {job_title} prep mock. "
                "We'll start with a strength, then areas to refine."
            ),
            jd_summary_short=" ".join(raw_jd.split())[:300],
            strong_probed=s_req if strong else "",
            weak_probed=w_req,
            gap_probed=g_req if gap else "",
            segments=segments,
        )

    async def _polish_with_llm(
        self,
        base: InterviewPlan,
        job_title: str,
        raw_jd: str,
        raw_resume: str,
        responsibilities: List[str],
        strong: Optional[dict],
        weak: Optional[dict],
        gap: Optional[dict],
    ) -> Optional[InterviewPlan]:
        """Ask LLM to shorten opening and questions; segment structure must stay the same."""
        excerpt = resume_excerpt(raw_resume)
        resp_text = "\n".join(f"- {r}" for r in responsibilities[:5])
        seg_desc = "\n".join(
            f"{i+1}. phase={s.phase} requirement={s.requirement!r} question={s.question!r}"
            for i, s in enumerate(base.segments)
        )

        prompt = (
            "You are designing a 5-minute CANDIDATE PREP mock interview (coaching, not hiring).\n"
            "Do NOT change segment count, order, or phase labels.\n"
            "Only refine opening_line, jd_summary_short, and each segment question (max 20 words).\n"
            "Keep bridges thematic. Use coaching tone on gap/weak segments.\n\n"
            f"JOB: {job_title}\nJD excerpt:\n{raw_jd[:1500]}\n\n"
            f"RESPONSIBILITIES:\n{resp_text or 'N/A'}\n\n"
            f"RESUME EXCERPT:\n{excerpt}\n\n"
            f"STRONG:\n{_format_topic_block(strong) if strong else 'None'}\n\n"
            f"WEAK:\n{_format_topic_block(weak) if weak else 'None'}\n\n"
            f"GAP:\n{_format_topic_block(gap) if gap else 'None'}\n\n"
            f"SEGMENTS (keep phases and requirements identical):\n{seg_desc}\n\n"
            "Return InterviewPlan with same segment phases and requirements."
        )

        result = await self.structured_llm.ainvoke([HumanMessage(content=prompt)])
        if not result:
            return None
        # Preserve deterministic requirements if LLM drifted
        for i, seg in enumerate(base.segments):
            if i < len(result.segments):
                result.segments[i].phase = seg.phase
                result.segments[i].requirement = seg.requirement
        result.strong_probed = base.strong_probed
        result.weak_probed = base.weak_probed
        result.gap_probed = base.gap_probed
        return result
