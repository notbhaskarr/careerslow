import asyncio
from typing import Optional
from langgraph.graph import StateGraph, START, END
from src.graph.state import GraphState
from src.parser.document_parser import DocumentParser
from src.db.qdrant_client import VectorDatabase, DUTY_RETRIEVAL_SECTIONS, SKILL_RETRIEVAL_SECTIONS
from src.generator.gap_generator import GapGenerator
from src.schemas.document_schemas import GapAnalysis, cap_importance_weight
from langchain_core.runnables.config import RunnableConfig
from src.db.redis_cache import RedisCache
import hashlib
from src.utils.pipeline_debug import log_parsed_resume, log_chunks

# Overall fit: weighted blend of required skills, experience duties, and nice-to-have.
# Ratio 65 : 15 : 20 (sums to 100).
BLEND_REQUIRED = 0.65
BLEND_DUTY = 0.15
BLEND_NICE = 0.20


def _overall_fit_score(
    req_avg: float,
    req_weight: float,
    resp_avg: float,
    resp_weight: float,
    nice_avg: float,
    nice_weight: float,
) -> float:
    """Blend category averages; renormalize when a category is absent."""
    blend: list[tuple[float, float]] = []
    if req_weight > 0:
        blend.append((BLEND_REQUIRED, req_avg))
    if resp_weight > 0:
        blend.append((BLEND_DUTY, resp_avg))
    if nice_weight > 0:
        blend.append((BLEND_NICE, nice_avg))
    if not blend:
        return 0.0
    total_w = sum(w for w, _ in blend)
    return sum(w * s for w, s in blend) / total_w


class WorkflowBuilder:
    def __init__(self):
        """Initializes the workflow builder and all the required components."""
        self.parser = DocumentParser()
        self.db = VectorDatabase()
        self.generator = GapGenerator()
        self.cache = RedisCache()
        
    def parse_resume_node(self, state: GraphState):
        """Node 1a: Parses raw resume text into structured models with Redis caching."""
        print(f"Executing Node: parse_resume for {state['resume_id']}")
        
        cached_resume = self.cache.get_resume(state["resume_id"])
        if cached_resume:
            log_parsed_resume(state["resume_id"], cached_resume, from_cache=True)
            has_index = bool(self.db.list_indexed_chunks(state["resume_id"], limit=1))
            return {"parsed_resume": cached_resume, "skip_indexing": has_index}
            
        print("Cache MISS. Parsing Resume via LLM...")
        parsed_res = self.parser.parse_resume(state["raw_resume"])
        parsed_res.raw_text = state["raw_resume"]
        log_parsed_resume(state["resume_id"], parsed_res, from_cache=False)
        self.cache.set_resume(state["resume_id"], parsed_res)
        return {"parsed_resume": parsed_res, "skip_indexing": False}

    def parse_jd_node(self, state: GraphState):
        """Node 1b: Parses raw JD text into structured Pydantic models with Redis caching."""
        print("Executing Node: parse_jd")
        
        # 1. Create a unique hash for the normalized Job Description text
        normalized_jd = " ".join(state["raw_jd"].split()).strip().lower()
        jd_hash = hashlib.md5(normalized_jd.encode("utf-8")).hexdigest()
        
        # 2. Check Redis Cache
        cached_jd = self.cache.get_jd(jd_hash)
        if cached_jd:
            return {"parsed_jd": cached_jd}
            
        # 3. Cache Miss: Run the LLM
        print("Cache MISS. Parsing JD via LLM...")
        parsed_jd = self.parser.parse_jd(state["raw_jd"])
        parsed_jd.raw_text = state["raw_jd"] 
        
        # 4. Save to Redis for future candidates
        self.cache.set_jd(jd_hash, parsed_jd)
        
        return {"parsed_jd": parsed_jd}
        
    def index_resume(self, state: GraphState):
        """Node 2: Embeds the parsed resume into Qdrant for Hybrid Search."""
        resume_id = state["resume_id"]
        if state.get("skip_indexing"):
            print("Skipping Qdrant Indexing (Vector chunks already exist in database).")
            existing = self.db.list_indexed_chunks(resume_id)
            if existing:
                chunk_debug = log_chunks(resume_id, existing, from_existing_index=True)
                self.cache.set_chunks(resume_id, chunk_debug)
                return {}
            print("skip_indexing set but Qdrant empty — re-indexing.")

        print("Executing Node: index_resume")
        staged = self.db.index_resume(
            parsed_resume=state["parsed_resume"],
            resume_id=resume_id,
        )
        chunk_debug = log_chunks(resume_id, staged, from_existing_index=False)
        self.cache.set_chunks(resume_id, chunk_debug)
        return {}
        
    async def generate_gap_analysis(self, state: GraphState, config: RunnableConfig):
        """Node 3: Retrieves evidence and generates gap analysis for skills and responsibility themes."""
        print("Executing Node: generate_gap_analysis (Parallelized)")
        analyses = []
        errors = []

        parsed_jd = state["parsed_jd"]
        required_reqs = parsed_jd.required_skills
        nice_to_have_reqs = parsed_jd.nice_to_have_skills
        responsibility_themes = getattr(parsed_jd, "responsibility_themes", None) or []

        semaphore = asyncio.Semaphore(5)

        async def process_item(
            label: str,
            weight: int,
            category: str,
            requirement_kind: str = "skill",
            section_types: Optional[list] = None,
        ):
            try:
                evidence = self.db.retrieve_evidence(
                    query=label,
                    resume_id=state["resume_id"],
                    section_types=section_types,
                )
            except Exception as e:
                return {"error": f"Database retrieval failed for '{label}': {str(e)}", "analysis": None}

            if not evidence:
                analysis = GapAnalysis(
                    category=category,
                    requirement=label,
                    match_score=2,
                    reasoning="Semantic similarity threshold not met. No relevant evidence found in the resume.",
                    gap_description=f"The candidate's resume lacks any semantically relevant evidence for {label}.",
                    tailored_bullets=[],
                )
            else:
                async with semaphore:
                    print(f"Analyzing Gap Score asynchronously for: {label} ({category})")
                    analysis = await self.generator.analyze_gap_score(
                        requirement=label,
                        evidence=evidence,
                        job_title=parsed_jd.job_title,
                        requirement_kind=requirement_kind,
                        config=config,
                    )
                    analysis.category = category
                    analysis.requirement = label

            return {"error": None, "analysis": analysis, "weight": weight, "category": category}

        tasks = []
        for req in required_reqs:
            tasks.append(
                process_item(
                    req.skill_name,
                    cap_importance_weight(req.importance_weight),
                    "required_skill",
                    section_types=SKILL_RETRIEVAL_SECTIONS,
                )
            )
        for req in nice_to_have_reqs:
            tasks.append(
                process_item(
                    req.skill_name,
                    cap_importance_weight(req.importance_weight),
                    "nice_to_have_skill",
                    section_types=SKILL_RETRIEVAL_SECTIONS,
                )
            )
        for theme in responsibility_themes:
            tasks.append(
                process_item(
                    theme.theme_name,
                    cap_importance_weight(theme.importance_weight),
                    "responsibility",
                    requirement_kind="responsibility",
                    section_types=DUTY_RETRIEVAL_SECTIONS,
                )
            )

        results = await asyncio.gather(*tasks)

        req_score, req_weight = 0.0, 0
        nice_score, nice_weight = 0.0, 0
        resp_score, resp_weight = 0.0, 0

        for res in results:
            if res["error"]:
                errors.append(res["error"])
            elif res["analysis"]:
                if res["analysis"].match_score == 0:
                    errors.append(f"LLM generation failed for requirement: {res['analysis'].requirement}")
                else:
                    cat = res["category"]
                    if cat == "required_skill":
                        req_score += res["analysis"].match_score * res["weight"]
                        req_weight += res["weight"]
                    elif cat == "nice_to_have_skill":
                        nice_score += res["analysis"].match_score * res["weight"]
                        nice_weight += res["weight"]
                    elif cat == "responsibility":
                        resp_score += res["analysis"].match_score * res["weight"]
                        resp_weight += res["weight"]
                analyses.append(res["analysis"])

        req_avg = (req_score / req_weight) if req_weight > 0 else 0.0
        nice_avg = (nice_score / nice_weight) if nice_weight > 0 else 0.0
        resp_avg = (resp_score / resp_weight) if resp_weight > 0 else 0.0

        score_breakdown = {
            "required_skills_avg": round(req_avg, 2),
            "responsibility_avg": round(resp_avg, 2),
            "nice_to_have_avg": round(nice_avg, 2),
        }

        overall_score = _overall_fit_score(
            req_avg, req_weight, resp_avg, resp_weight, nice_avg, nice_weight
        )

        gaps_for_bullets = [a for a in analyses if a.match_score < 8]

        if gaps_for_bullets:
            batch_bullets_response = await self.generator.generate_batch_bullets(
                partial_gaps=gaps_for_bullets,
                raw_jd=state["raw_jd"],
                raw_resume=state["raw_resume"],
                config=config,
            )

            suggestions_map = {
                res.requirement.lower().strip(): res.suggestions
                for res in batch_bullets_response.results
            }
            for analysis in analyses:
                req_key = analysis.requirement.lower().strip()
                matched = suggestions_map.get(req_key)
                if not matched:
                    for k, v in suggestions_map.items():
                        if k in req_key or req_key in k:
                            matched = v
                            break
                if matched:
                    analysis.resume_suggestions = matched
                    analysis.tailored_bullets = [s.text for s in matched]

        return {
            "gap_analyses": analyses,
            "overall_fit_score": round(overall_score, 2),
            "score_breakdown": score_breakdown,
            "errors": errors,
        }
        
    def build_graph(self):
        """Compiles the LangGraph with parallel parsing optimization."""
        workflow = StateGraph(GraphState)
        
        # Add Nodes
        workflow.add_node("parse_resume", self.parse_resume_node)
        workflow.add_node("parse_jd", self.parse_jd_node)
        
        # Dummy synchronization node
        def join_node(state):
            print("Executing Node: Synchronization (Join)")
            return {}
        workflow.add_node("join_parsing", join_node)
        
        workflow.add_node("index_resume", self.index_resume)
        workflow.add_node("generate_gap_analysis", self.generate_gap_analysis)
        
        # Define Edges (Parallel Parsing -> Sync -> Linear)
        workflow.add_edge(START, "parse_resume")
        workflow.add_edge(START, "parse_jd")
        
        # Both parallel nodes feed into the join node
        workflow.add_edge("parse_resume", "join_parsing")
        workflow.add_edge("parse_jd", "join_parsing")
        
        # After both are parsed, we index the resume and generate the analysis
        workflow.add_edge("join_parsing", "index_resume")
        workflow.add_edge("index_resume", "generate_gap_analysis")
        workflow.add_edge("generate_gap_analysis", END)
        
        return workflow.compile()
