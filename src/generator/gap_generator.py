import os
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional, List
from src.schemas.document_schemas import GapAnalysis, BatchBulletsResponse

class GapGenerator:
    def __init__(self):
        """
        Initializes the GapGenerator with two Gemini models for Decomposed Generation.
        """
        # Step 1: Scorer LLM (Temp = 0.0 for deterministic math)
        self.scorer_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.0)
        self.structured_scorer = self.scorer_llm.with_structured_output(GapAnalysis)
        
        # Step 2: Writer LLM (Temp = 0.4 for concise, realistic writing)
        self.writer_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.4)
        self.structured_writer = self.writer_llm.with_structured_output(BatchBulletsResponse)

    async def analyze_gap_score(
        self,
        requirement: str,
        evidence: list,
        job_title: str = "Software Engineer",
        requirement_kind: str = "skill",
        config: Optional[dict] = None,
    ) -> GapAnalysis:
        """
        Step 1: Asynchronously generates the deterministic math score and reasoning.

        requirement_kind: 'skill' for KSA/tools, 'responsibility' for JD duty themes.
        """
        evidence_text = "\n".join(f"- {e}" for e in evidence) if evidence else "NO EVIDENCE FOUND."

        if requirement_kind == "responsibility":
            kind_context = (
                "This is a JOB DUTY / EXPERIENCE ALIGNMENT check (not a named tool requirement).\n"
                "Score based on whether resume experience bullets show the candidate has DONE similar work.\n"
                "Score 9-10: Clear, direct experience performing this duty or very close equivalent.\n"
                "Score 5-8: Partial or adjacent experience (related testing/QA work but not exact duty wording).\n"
                "Score 1-4: No relevant experience evidence.\n"
            )
        else:
            kind_context = (
                "This is a SKILLS / TOOLS requirement check.\n"
                "Score 9-10: Exact skill/tool explicitly named.\n"
                "Score 5-8: Adjacent or partial knowledge.\n"
                "Score 1-4: Missing or extremely weak evidence.\n"
            )

        prompt = (
            f"You are an expert technical career coach and resume reviewer helping a candidate apply for a '{job_title}' role.\n"
            "Your task is to analyze how well a candidate meets a specific job requirement based ONLY on the provided evidence extracted from their resume.\n\n"
            f"{kind_context}\n"
            "RULES & RUBRIC:\n"
            "1. Write out your step-by-step `reasoning`.\n"
            "2. Assign the `match_score` based on the rubric above.\n"
            "   - OR / ANY-OF RULE (skills only): If the requirement lists interchangeable options, score 9-10 if ANY is evidenced.\n"
            "   - Do not confuse similar but distinct skills (e.g. C vs C#).\n"
            "3. If 'NO EVIDENCE FOUND' is provided, the match score must be 1-3.\n"
            "4. Write a brief 'gap_description' explaining what is missing or weak.\n"
            "5. Leave the 'tailored_bullets' array empty. It will be generated later.\n\n"
            f"Job Requirement:\n{requirement}\n\n"
            f"Candidate Evidence:\n{evidence_text}"
        )
        
        try:
            result = await self.structured_scorer.ainvoke(prompt, config=config)
            return result
        except Exception as e:
            print(f"Warning: Async LLM Scoring failed for requirement '{requirement[:20]}...'. Error: {str(e)}")
            category = "responsibility" if requirement_kind == "responsibility" else "required_skill"
            return GapAnalysis(
                category=category,
                requirement=requirement,
                reasoning="System Error: Analysis failed. Unable to evaluate reasoning.",
                match_score=0,
                gap_description="System Error: LLM analysis failed due to timeout or parsing issue. Please review manually.",
                tailored_bullets=[]
            )

    async def generate_batch_bullets(self, partial_gaps: List[GapAnalysis], raw_jd: str, raw_resume: str, config: Optional[dict] = None) -> BatchBulletsResponse:
        """
        Step 2: Batch generation of highly creative, jargon-free tailored bullets.
        Passes the entire original JD and Resume to avoid Context Disconnect.
        """
        print("Executing Step 2: Batch Generating Tailored Bullets (Temp=0.4)...")
        
        gaps_summary = ""
        for i, gap in enumerate(partial_gaps):
            gaps_summary += f"[{i+1}] Requirement: {gap.requirement}\nGap: {gap.gap_description}\nScore: {gap.match_score}/10\n\n"

        prompt = (
            "You are an elite Resume Editor helping a candidate tighten their resume for a job — not inflate it.\n"
            "For each gap below, return 1-2 suggestions. The candidate should keep resume length roughly the same.\n\n"
            "CRITICAL RULES:\n"
            "1. PREFER EDIT OVER ADD: First look for an existing experience bullet, project line, or skills line that can "
            "be revised to surface the missing keyword or duty. Use action='edit' and quote that line in target_line.\n"
            "2. ADD SPARINGLY: Use action='add' only when no existing line can reasonably absorb the gap. At most ONE add per requirement.\n"
            "3. STRICTLY FACTUAL: Do not invent skills, tools, or projects absent from the original resume.\n"
            "4. SHORT: Each text field under 20 words, resume-ready, strong action verbs, no buzzwords (Leveraged, Spearheaded).\n"
            "5. COMPLETE MISMATCH: If they truly lack a tool, edit the closest real bullet to highlight adjacent depth — do not fabricate the missing tool.\n"
            "6. INTEGRATE, DON'T STACK: When editing, merge JD keywords into the existing bullet rather than writing a second parallel bullet.\n\n"
            f"--- ORIGINAL JOB DESCRIPTION ---\n{raw_jd}\n\n"
            f"--- ORIGINAL RESUME ---\n{raw_resume}\n\n"
            f"--- GAPS TO ADDRESS ---\n{gaps_summary}\n\n"
            "Task: Return each requirement with 1-2 suggestions (action, target_line when edit, text)."
        )

        try:
            return await self.structured_writer.ainvoke(prompt, config=config)
        except Exception as e:
            print(f"Warning: Batch Bullet generation failed. Error: {str(e)}")
            return BatchBulletsResponse(results=[])
