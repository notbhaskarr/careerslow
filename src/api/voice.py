"""Voice WebSocket routes and post-interview candidate debrief extraction."""

import asyncio
import logging
import os
from typing import List

from fastapi import APIRouter, HTTPException, WebSocket
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.api.extraction_helpers import (
    build_study_topics,
    heuristic_debrief,
    parse_extraction_json,
    truncate_transcript,
)
from src.voice.interview_service import InterviewService
from src.voice.llm_utils import ainvoke_with_retry

logger = logging.getLogger(__name__)
voice_router = APIRouter()
cache = None


class DebriefResult(BaseModel):
    """Candidate-facing mock interview debrief."""

    strong_topic: str = Field(default="")
    strong_assessment: str = Field(default="")
    weak_topic: str = Field(default="")
    weak_assessment: str = Field(default="")
    gap_topic: str = Field(default="")
    gap_assessment: str = Field(default="")
    communication_notes: str = Field(default="")
    overall_readiness: str = Field(default="")
    study_topics: List[str] = Field(default_factory=list)


def _apply_session_meta(result: DebriefResult, session_meta: dict) -> DebriefResult:
    """Clear topic fields that were not reached in the live session."""
    if not session_meta:
        return result
    if not session_meta.get("strong_probed") and not session_meta.get("strength_probed"):
        result.strong_topic = ""
        result.strong_assessment = ""
    if not session_meta.get("weak_probed"):
        result.weak_topic = ""
        result.weak_assessment = ""
    if not session_meta.get("gap_probed"):
        result.gap_topic = ""
        result.gap_assessment = ""
    return result


def _load_extraction_prompt_template() -> str:
    path = "extraction_prompt.txt"
    if not os.path.exists(path):
        return ""
    with open(path, "r") as f:
        content = f.read()
    marker = "*End of Voice Prompt*"
    if marker in content:
        return content.split(marker, 1)[-1]
    return content


def _build_debrief_prompt(
    template: str,
    *,
    job_title: str,
    jd_summary: str,
    session_meta: dict,
    plan: dict,
    transcript: str,
) -> str:
    strong = session_meta.get("strong_probed") or session_meta.get("strength_probed", "")
    weak = session_meta.get("weak_probed", "")
    gap = session_meta.get("gap_probed", "")
    segments_asked = session_meta.get("segments_asked", [])

    if template:
        return (
            template.replace("{{job_title}}", job_title)
            .replace("{{job_description_summary}}", jd_summary)
            .replace("{{strength_from_log_selection_or_empty}}", strong)
            .replace("{{gap_from_log_selection_or_empty}}", gap)
            .replace("{{weak_from_log_selection_or_empty}}", weak)
            .replace("{{full_strengths_list}}", plan.get("strong_probed", ""))
            .replace("{{full_gaps_list}}", plan.get("gap_probed", ""))
            .replace("{{segments_asked}}", str(segments_asked))
            .replace("{{full_call_transcript}}", truncate_transcript(transcript))
        )

    return f"""
You are a career coach debriefing a candidate after a mock interview prep session.
Write coaching feedback — not hiring-manager evaluation.

Job: {job_title}
Summary: {jd_summary}
Strong topic reached: {strong or '(not reached)'}
Weak topic reached: {weak or '(not reached)'}
Gap topic reached: {gap or '(not reached)'}
Segments reached: {segments_asked}

TRANSCRIPT:
{truncate_transcript(transcript)}

Return JSON with keys:
strong_topic, strong_assessment, weak_topic, weak_assessment,
gap_topic, gap_assessment, communication_notes, overall_readiness, study_topics (array of strings).
Leave fields empty if that topic was not discussed. study_topics: 3-6 specific things to read/practice.
"""


async def _extract_debrief(
    prompt: str, transcript: str, session_meta: dict, gap_analyses: list
) -> DebriefResult:
    """Try structured LLM, JSON fallback, then heuristic debrief."""
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

    try:
        structured = llm.with_structured_output(DebriefResult)
        result = await ainvoke_with_retry(structured, [HumanMessage(content=prompt)])
        if result:
            if not result.study_topics:
                result.study_topics = build_study_topics(session_meta, gap_analyses)
            return _apply_session_meta(result, session_meta)
    except Exception as e:
        logger.warning(f"Structured debrief failed: {e}")

    try:
        raw = await ainvoke_with_retry(
            llm,
            [HumanMessage(content=prompt + "\n\nRespond with JSON only, no markdown.")],
        )
        parsed = parse_extraction_json(raw.content if raw else "")
        if parsed:
            parsed.setdefault("study_topics", build_study_topics(session_meta, gap_analyses))
            result = DebriefResult.model_validate(parsed)
            return _apply_session_meta(result, session_meta)
    except Exception as e:
        logger.warning(f"JSON debrief fallback failed: {e}")

    logger.warning("Using heuristic debrief fallback")
    return DebriefResult.model_validate(
        heuristic_debrief(transcript, session_meta, gap_analyses)
    )


@voice_router.websocket("/api/interview/ws/{pair_id}")
async def interview_websocket(websocket: WebSocket, pair_id: str):
    """Live voice mock interview for a resume+JD pair."""
    await websocket.accept()

    if not cache:
        await websocket.close(code=1011, reason="Cache not initialized")
        return

    service = InterviewService(cache)
    try:
        session = await service.create_session(websocket, pair_id)
    except ValueError as e:
        logger.error(str(e))
        await websocket.close(code=1008, reason=str(e))
        return

    await session.start()
    await session.run()


@voice_router.post("/api/interview/debrief/{session_id}", response_model=DebriefResult)
async def debrief_interview(session_id: str):
    """
    Generate candidate debrief after a voice session ends.

    Args:
        session_id: Returned in session_started WebSocket message.

    Returns:
        DebriefResult with per-topic assessments and study_topics list.
    """
    if not cache:
        raise HTTPException(status_code=500, detail="Cache not initialized")

    cached = cache.get_debrief(session_id)
    if cached:
        return DebriefResult.model_validate(cached)

    transcript = ""
    pair_id = None
    for _ in range(6):
        transcript = cache.get_transcript(session_id)
        pair_id = cache.get_session_pair(session_id)
        if transcript and pair_id:
            break
        await asyncio.sleep(0.5)

    if not transcript:
        raise HTTPException(status_code=404, detail="No transcript found for this session")

    if not pair_id:
        raise HTTPException(status_code=404, detail="Session not linked to an analysis pair")

    analysis = cache.get_analysis(pair_id) or {}
    plan = cache.get_interview_plan(pair_id) or {}
    session_meta = cache.get_session_meta(session_id) or {}
    gap_analyses = analysis.get("gap_analyses", [])

    job_title = analysis.get("job_title", "Software Engineer")
    jd_summary = plan.get("jd_summary_short") or analysis.get("jd_summary_short", "")

    template = _load_extraction_prompt_template()
    prompt = _build_debrief_prompt(
        template,
        job_title=job_title,
        jd_summary=jd_summary,
        session_meta=session_meta,
        plan=plan,
        transcript=transcript,
    )

    try:
        result = await _extract_debrief(prompt, transcript, session_meta, gap_analyses)
        cache.set_debrief(session_id, result.model_dump())
        return result
    except Exception as e:
        logger.error(f"Debrief failed: {e}")
        result = DebriefResult.model_validate(
            heuristic_debrief(transcript, session_meta, gap_analyses)
        )
        cache.set_debrief(session_id, result.model_dump())
        return result


@voice_router.post("/api/interview/extract/{session_id}", response_model=DebriefResult)
async def extract_interview_notes_legacy(session_id: str):
    """Alias for debrief endpoint (backward compatible path)."""
    return await debrief_interview(session_id)
