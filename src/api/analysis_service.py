"""Shared resume+JD analysis orchestration for API endpoints."""

import logging
from typing import Any, Optional

from src.generator.interview_plan_generator import InterviewPlanGenerator
from src.graph.state import GraphState
from src.graph.workflow import WorkflowBuilder
from src.utils.cache_keys import make_jd_id, make_pair_id, make_resume_id
from src.utils.pipeline_debug import log_extraction, log_parsed_resume
from src.utils.text import resume_excerpt

logger = logging.getLogger(__name__)


async def run_analysis(
    workflow_builder: WorkflowBuilder,
    graph,
    resume_text: str,
    jd_text: str,
    *,
    source_filename: str = "draft.txt",
    force_refresh: bool = False,
    previous_overall_fit_score: Optional[float] = None,
) -> dict[str, Any]:
    """
    Run full gap-analysis pipeline for resume text + JD.

    Returns response dict suitable for AnalysisResponse and Redis cache.
    """
    if not resume_text.strip():
        raise ValueError("Resume text is empty.")
    if not jd_text.strip():
        raise ValueError("Job description is empty.")

    resume_id = make_resume_id(resume_text)
    jd_id = make_jd_id(jd_text)
    pair_id = make_pair_id(resume_text, jd_text)

    extract_payload = log_extraction(
        resume_id, source_filename, [resume_text], resume_text
    )
    workflow_builder.cache.set_extraction(resume_id, extract_payload)

    if not force_refresh:
        cached = workflow_builder.cache.get_analysis(pair_id)
        if cached and cached.get("gap_analyses"):
            plan = workflow_builder.cache.get_interview_plan(pair_id)
            if plan:
                logger.info("Cache HIT for pair %s", pair_id)
                cached_parsed = workflow_builder.cache.get_resume(resume_id)
                if cached_parsed:
                    log_parsed_resume(resume_id, cached_parsed, from_cache=True)
                out = dict(cached)
                out["resume_id"] = resume_id
                out["jd_id"] = jd_id
                out["pair_id"] = pair_id
                out["resume_text"] = resume_text
                out["cached"] = True
                if previous_overall_fit_score is not None:
                    out["previous_overall_fit_score"] = previous_overall_fit_score
                return out

    initial_state = GraphState(
        resume_id=resume_id,
        raw_resume=resume_text,
        raw_jd=jd_text,
        parsed_resume=None,
        parsed_jd=None,
        gap_analyses=[],
        overall_fit_score=0.0,
        score_breakdown={},
        errors=[],
        skip_indexing=False,
        retry_count=0,
        is_grounded=True,
        session_id=pair_id,
    )

    final_state = await graph.ainvoke(initial_state)

    response_data: dict[str, Any] = {
        "resume_id": resume_id,
        "jd_id": jd_id,
        "pair_id": pair_id,
        "overall_fit_score": final_state["overall_fit_score"],
        "score_breakdown": final_state.get("score_breakdown", {}),
        "gap_analyses": [gap.model_dump() for gap in final_state["gap_analyses"]],
        "errors": final_state["errors"],
        "job_title": final_state["parsed_jd"].job_title if final_state.get("parsed_jd") else "Unknown Role",
        "raw_jd_summary": final_state["parsed_jd"].raw_text if final_state.get("parsed_jd") else "",
        "resume_excerpt": resume_excerpt(resume_text),
        "resume_text": resume_text,
        "cached": False,
    }
    if previous_overall_fit_score is not None:
        response_data["previous_overall_fit_score"] = previous_overall_fit_score

    workflow_builder.cache.set_analysis(pair_id, response_data)

    try:
        plan_generator = InterviewPlanGenerator()
        responsibilities = (
            final_state["parsed_jd"].responsibilities if final_state.get("parsed_jd") else []
        )
        interview_plan = await plan_generator.generate(
            gap_analyses=response_data["gap_analyses"],
            job_title=response_data["job_title"],
            raw_jd=jd_text,
            raw_resume=resume_text,
            responsibilities=responsibilities,
        )
        plan_dict = interview_plan.model_dump()
        plan_dict["strength_probed"] = plan_dict.get("strong_probed", "")
        workflow_builder.cache.set_interview_plan(pair_id, plan_dict)
        response_data["jd_summary_short"] = interview_plan.jd_summary_short
        workflow_builder.cache.set_analysis(pair_id, response_data)
    except Exception as e:
        logger.error("Interview plan generation failed: %s", e)

    return response_data
