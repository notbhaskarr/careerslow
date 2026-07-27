from src.utils.text import extract_candidate_name
"""Creates voice interview sessions from cached pair analysis."""

import logging
import uuid

from fastapi import WebSocket

from src.db.redis_cache import RedisCache
from src.voice.prompt_builder import build_system_prompt
from src.voice.session import InterviewSession

logger = logging.getLogger(__name__)


class InterviewService:
    def __init__(self, cache: RedisCache):
        self.cache = cache

    def load_context(self, pair_id: str) -> dict:
        """
        Load analysis and interview plan for a resume+JD pair.

        Args:
            pair_id: Composite key from make_pair_id().

        Raises:
            ValueError: If analysis or plan missing (user must analyze first).
        """
        analysis = self.cache.get_analysis(pair_id)
        if not analysis:
            raise ValueError("No analysis found. Run resume analysis first.")

        plan = self.cache.get_interview_plan(pair_id)
        if not plan:
            raise ValueError("No interview plan found. Re-run resume analysis.")

        return {
            "analysis": analysis,
            "plan": plan,
            "job_title": analysis.get("job_title", "Software Engineer"),
            "resume_excerpt": analysis.get("resume_excerpt", ""),
            "jd_summary": plan.get("jd_summary_short") or analysis.get("jd_summary_short", ""),
        }

    async def create_session(self, websocket: WebSocket, pair_id: str) -> InterviewSession:
        """
        Build an InterviewSession wired to Redis and the interview plan.

        Returns:
            InterviewSession with a fresh session_id for transcript/debrief storage.
        """
        ctx = self.load_context(pair_id)
        session_id = f"sess_{uuid.uuid4().hex[:16]}"
        analysis = ctx["analysis"]
        plan = dict(ctx["plan"])
        plan.setdefault("job_title", ctx["job_title"])
        name = analysis.get("candidate_name") or extract_candidate_name(analysis.get("resume_excerpt", ""))
        plan.setdefault("candidate_name", name)
        system_prompt = build_system_prompt(
            ctx["job_title"],
            ctx["jd_summary"],
            ctx["resume_excerpt"],
        )
        return InterviewSession(
            websocket=websocket,
            pair_id=pair_id,
            session_id=session_id,
            cache=self.cache,
            system_prompt=system_prompt,
            plan=plan,
        )
