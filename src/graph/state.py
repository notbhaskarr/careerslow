from typing import TypedDict, List, Optional
from src.schemas.document_schemas import ParsedResume, ParsedJD, GapAnalysis

class GraphState(TypedDict):
    """
    The State object that is passed between LangGraph nodes.
    It holds the raw inputs, the parsed intermediate models, and the final output.
    """
    # Inputs
    raw_resume: str
    raw_jd: str
    resume_id: str
    
    # Intermediate State
    parsed_resume: Optional[ParsedResume]
    parsed_jd: Optional[ParsedJD]
    skip_indexing: bool
    
    # Final Output
    gap_analyses: List[GapAnalysis]
    overall_fit_score: float
    score_breakdown: Optional[dict]
    
    # Phase 3 Loop Control & Observability
    retry_count: int
    is_grounded: bool
    errors: List[str]
    session_id: str
