import logging
import os
from typing import List, Optional

import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.graph.workflow import WorkflowBuilder
from src.api.analysis_service import run_analysis
from src.utils.cache_keys import make_jd_id, make_pair_id, make_resume_id
from src.utils.pipeline_debug import log_parsed_resume, parsed_resume_summary
from src.utils.resume_draft import AcceptedEdit, apply_edits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CareersLow AI Interviewer API",
    description="Resume/JD gap analysis, resume bullet suggestions, and mock interview prep.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

workflow_builder = WorkflowBuilder()
graph = workflow_builder.build_graph()

from src.api.voice import voice_router
import src.api.voice as voice_module

voice_module.cache = workflow_builder.cache
app.include_router(voice_router)


class AnalysisResponse(BaseModel):
    resume_id: str
    jd_id: str
    pair_id: str
    overall_fit_score: float
    score_breakdown: dict = {}
    gap_analyses: list
    errors: list
    cached: bool = False
    resume_text: str = ""
    previous_overall_fit_score: Optional[float] = None


class SaveAndAnalyzeRequest(BaseModel):
    jd_text: str
    resume_text: str
    accepted_edits: List[AcceptedEdit] = []
    previous_overall_fit_score: Optional[float] = None


@app.get("/health")
async def health():
    """Render / uptime health check."""
    return {"status": "ok"}


@app.post("/api/analyze-resume", response_model=AnalysisResponse)
async def analyze_resume(
    resume_pdf: UploadFile = File(...),
    jd_text: str = Form(...),
    force_refresh: bool = Query(False, description="Bypass cached analysis for this resume+JD pair"),
):
    """
    Parse resume PDF + JD, run gap analysis, generate interview plan.

    Re-uploading a revised resume produces a new resume_id (content hash).
    Same resume against a different JD produces a different pair_id.
    """
    logger.info(f"Received analysis request for file: {resume_pdf.filename}")

    if not resume_pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        pdf_bytes = await resume_pdf.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_texts = [page.get_text() for page in doc]
        resume_text = "".join(page_texts)
        if not resume_text.strip():
            raise ValueError("Extracted text is empty. The PDF might be scanned images.")
    except Exception as e:
        logger.error(f"Failed to parse PDF: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail="Failed to parse the uploaded PDF file. Ensure it is a valid, text-based PDF.",
        )

    resume_id = make_resume_id(resume_text)
    jd_id = make_jd_id(jd_text)
    pair_id = make_pair_id(resume_text, jd_text)
    logger.info(f"IDs: resume={resume_id} jd={jd_id} pair={pair_id}")

    try:
        response_data = await run_analysis(
            workflow_builder,
            graph,
            resume_text,
            jd_text,
            source_filename=resume_pdf.filename or "upload.pdf",
            force_refresh=force_refresh,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"LangGraph execution failed: {e}")
        raise HTTPException(status_code=500, detail="AI processing failed on the server. Please try again.")

    logger.info(f"Processed pair {response_data['pair_id']}. Score: {response_data['overall_fit_score']}")

    return AnalysisResponse(
        resume_id=response_data["resume_id"],
        jd_id=response_data["jd_id"],
        pair_id=response_data["pair_id"],
        overall_fit_score=response_data["overall_fit_score"],
        score_breakdown=response_data.get("score_breakdown", {}),
        gap_analyses=response_data["gap_analyses"],
        errors=response_data["errors"],
        cached=response_data.get("cached", False),
        resume_text=response_data.get("resume_text", resume_text),
        previous_overall_fit_score=response_data.get("previous_overall_fit_score"),
    )


@app.post("/api/save-and-analyze", response_model=AnalysisResponse)
async def save_and_analyze(body: SaveAndAnalyzeRequest):
    """
    Apply accepted resume edits, then run a full match analysis once.

    Used after the candidate finalizes draft changes in the UI.
    """
    revised = apply_edits(body.resume_text, body.accepted_edits)
    if not revised.strip():
        raise HTTPException(status_code=400, detail="Revised resume text is empty.")

    try:
        response_data = await run_analysis(
            workflow_builder,
            graph,
            revised,
            body.jd_text,
            source_filename="revised_draft.txt",
            force_refresh=True,
            previous_overall_fit_score=body.previous_overall_fit_score,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Save-and-analyze failed: {e}")
        raise HTTPException(status_code=500, detail="AI processing failed on the server. Please try again.")

    return AnalysisResponse(
        resume_id=response_data["resume_id"],
        jd_id=response_data["jd_id"],
        pair_id=response_data["pair_id"],
        overall_fit_score=response_data["overall_fit_score"],
        score_breakdown=response_data.get("score_breakdown", {}),
        gap_analyses=response_data["gap_analyses"],
        errors=response_data["errors"],
        cached=False,
        resume_text=response_data.get("resume_text", revised),
        previous_overall_fit_score=response_data.get("previous_overall_fit_score"),
    )


@app.get("/api/debug/pipeline")
async def debug_pipeline(resume_id: str = Query(..., description="resume_id from analyze response")):
    """
    Return PDF extraction + structured parse for QA after one analyze run.

    Extraction: extract:{resume_id} in Redis + eval/debug_runs/{resume_id}_extracted.txt
    Parse: resume:{resume_id} in Redis + eval/debug_runs/{resume_id}_parsed.json
    Chunks: chunks:{resume_id} in Redis + eval/debug_runs/{resume_id}_chunks.json
    """
    extraction = workflow_builder.cache.get_extraction(resume_id)
    parsed = workflow_builder.cache.get_resume(resume_id)
    chunks = workflow_builder.cache.get_chunks(resume_id)
    if not extraction and not parsed and not chunks:
        raise HTTPException(status_code=404, detail=f"No pipeline debug data for {resume_id}")

    out = {"resume_id": resume_id}
    if extraction:
        out["extraction"] = {
            "source_filename": extraction.get("source_filename"),
            "page_count": extraction.get("page_count"),
            "page_char_counts": extraction.get("page_char_counts"),
            "char_count": extraction.get("char_count"),
            "extracted_at": extraction.get("extracted_at"),
            "text": extraction.get("text"),
        }
    if parsed:
        out["parsed"] = parsed.model_dump()
        out["parsed_summary"] = parsed_resume_summary(parsed)
    if chunks:
        out["chunks"] = chunks
    elif parsed:
        # Fallback: load live from Qdrant if Redis chunks key missing
        live = workflow_builder.db.list_indexed_chunks(resume_id)
        if live:
            out["chunks"] = {
                "chunk_count": len(live),
                "from_existing_index": True,
                "chunks": [
                    {"index": i + 1, "section_type": c.get("section_type"), "text": c.get("text")}
                    for i, c in enumerate(live)
                ],
            }
    return out


@app.get("/api/debug/redis/pair")
async def debug_redis_pair(pair_id: str = Query(..., description="pair_id from analyze response")):
    """
    Inspect Redis cache keys for a resume+JD pair (analysis, plan, parsed docs).

    Used by the in-app Redis Debug panel during QA.
    """
    try:
        return workflow_builder.cache.inspect_pair(pair_id)
    except Exception as e:
        logger.error(f"Redis pair inspect failed: {e}")
        raise HTTPException(status_code=503, detail="Redis unavailable")


@app.get("/api/debug/redis/session")
async def debug_redis_session(session_id: str = Query(..., description="session_id from voice mock")):
    """
    Inspect Redis keys for a voice session (transcript, meta, debrief, pair link).
    """
    try:
        return workflow_builder.cache.inspect_session(session_id)
    except Exception as e:
        logger.error(f"Redis session inspect failed: {e}")
        raise HTTPException(status_code=503, detail="Redis unavailable")


@app.get("/api/debug/redis/current")
async def debug_redis_current(
    pair_id: str = Query(None),
    session_id: str = Query(None),
):
    """
    Convenience endpoint: pass pair_id and/or session_id from sessionStorage.
    """
    out = {}
    if pair_id:
        out["pair"] = workflow_builder.cache.inspect_pair(pair_id)
    if session_id:
        out["session"] = workflow_builder.cache.inspect_session(session_id)
    if not out:
        raise HTTPException(status_code=400, detail="Provide pair_id and/or session_id")
    return out


os.makedirs("frontend", exist_ok=True)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
