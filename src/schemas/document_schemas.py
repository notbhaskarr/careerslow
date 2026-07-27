from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

MAX_IMPORTANCE_WEIGHT = 9


def cap_importance_weight(weight: int) -> int:
    return max(1, min(int(weight), MAX_IMPORTANCE_WEIGHT))

class ParsedResume(BaseModel):
    """Schema for the structured extraction of a Resume."""
    raw_text: str = Field(default="", description="Injected by pipeline after parse; LLM should leave empty.")
    technical_skills: List[str] = Field(
        default_factory=list,
        description="Tools, languages, frameworks, methods from the resume. Each entry becomes one search chunk — "
        "preserve compound/grouped names; avoid bare ambiguous tokens.",
    )
    soft_skills: List[str] = Field(
        default_factory=list,
        description="Interpersonal traits only (leadership, communication). NOT methodologies like Agile/Scrum.",
    )
    experience_sections: List[str] = Field(
        default_factory=list,
        description="Complete standalone experience bullets as full sentences with tools and outcomes in context.",
    )
    projects: List[str] = Field(
        default_factory=list,
        description="Complete project descriptions as standalone sentences.",
    )
    domain_expertise: List[str] = Field(
        default_factory=list,
        description="Industries, domains, or business areas explicitly stated (e.g. Fintech, Telecom BSS).",
    )

class ResumeSuggestion(BaseModel):
    action: str = Field(
        description="'edit' to revise an existing resume line in place; 'add' only when no existing line can absorb the gap.",
    )
    target_line: str = Field(
        default="",
        description="When action=edit: exact or close quote of the existing resume bullet/skill line to update.",
    )
    text: str = Field(
        description="Revised wording (edit) or new bullet (add). Under 20 words, resume-ready.",
    )


class GapAnalysis(BaseModel):
    """
    Structured output for the gap analysis generation (Node 5).
    """
    category: str = Field(
        default="required_skill",
        description="required_skill, nice_to_have_skill, or responsibility",
    )
    requirement: str = Field(
        description="The specific job description requirement being analyzed."
    )
    reasoning: str = Field(
        description="Step-by-step reasoning evaluating the candidate's evidence against the requirement. MUST be generated before the match_score."
    )
    match_score: int = Field(
        description="A score from 1 to 10 evaluating how well the candidate's evidence matches this requirement."
    )
    gap_description: str = Field(
        description="A brief explanation of the gap between the candidate's experience and the requirement."
    )
    tailored_bullets: List[str] = Field(
        description="Legacy display bullets; prefer resume_suggestions.",
        default_factory=list,
    )
    resume_suggestions: List[ResumeSuggestion] = Field(
        default_factory=list,
        description="1-2 edit-in-place or add-only suggestions to close the gap without bloating the resume.",
    )

class ResponsibilityTheme(BaseModel):
    theme_name: str = Field(
        description="Search-optimized label for a JD duty theme using exact JD terms, under ~12 words. "
        "E.g. 'Exploratory regression E2E test execution'.",
    )
    importance_weight: int = Field(
        description="Importance 1-9 (max 9). Use JD time-allocation % when stated (e.g. 50% block → 9; 25% → 7-8). Default 8.",
    )

    @field_validator("importance_weight", mode="before")
    @classmethod
    def _cap_importance_weight(cls, value: int) -> int:
        return cap_importance_weight(value)


class JobRequirement(BaseModel):
    skill_name: str = Field(
        description="Search-optimized label using exact terms from the JD: lead with the primary keyword, "
        "keep under ~12 words, no filler like 'Experience with'. "
        "For OR groups: 'At least one of: [terms exactly as stated in JD]'."
    )
    importance_weight: int = Field(
        description="Criticality 1-9 (max 9). Required/must-have: 7-9. Preferred/nice-to-have: 3-6."
    )

    @field_validator("importance_weight", mode="before")
    @classmethod
    def _cap_importance_weight(cls, value: int) -> int:
        return cap_importance_weight(value)


class ParsedJD(BaseModel):
    """Schema for the structured extraction of a Job Description."""
    raw_text: str = Field(default="", description="Injected by pipeline after parse; LLM should leave empty.")
    job_title: str = Field(
        description="The title of the position (e.g., Senior Software Engineer). Default to 'the target role' if unknown.",
        default="the target role"
    )
    required_skills: List[JobRequirement] = Field(
        default_factory=list,
        description="Must-have skills/tools/methods from the JD. Target 5-12 total requirements across required + nice_to_have combined.",
    )
    nice_to_have_skills: List[JobRequirement] = Field(
        default_factory=list,
        description="Preferred/bonus skills from the JD. Hard skills and named tools only unless soft skill has searchable JD anchors.",
    )
    responsibility_themes: List[ResponsibilityTheme] = Field(
        default_factory=list,
        description="5-8 scorable duty themes from Principal Accountabilities / job duties. Each becomes an experience-alignment query.",
    )
    responsibilities: List[str] = Field(
        default_factory=list,
        description="Full day-to-day duty sentences for interview context (NOT individually scored).",
    )
    qualifications: List[str] = Field(default_factory=list, description="Extracted required qualifications (e.g. degrees, certifications)")
    domain_or_industry: List[str] = Field(default_factory=list, description="Extracted industries or domains (e.g. Fintech, Healthcare)")

class RequirementBullets(BaseModel):
    requirement: str = Field(description="The exact requirement text from the input")
    suggestions: List[ResumeSuggestion] = Field(
        description="1-2 resume edit/add suggestions; prefer edit over add.",
        default_factory=list,
    )
    tailored_bullets: List[str] = Field(
        default_factory=list,
        description="Deprecated; leave empty.",
    )

class BatchBulletsResponse(BaseModel):
    results: List[RequirementBullets] = Field(description="A list of tailored bullets for each requirement evaluated")

class InterviewSegment(BaseModel):
    phase: str = Field(description="strong, weak, gap, or close")
    requirement: str = Field(default="", description="JD requirement being discussed")
    match_score: int = Field(default=0)
    resume_evidence: str = Field(default="", description="Evidence from resume for this topic")
    gap_summary: str = Field(default="", description="What is missing for gap topics")
    question: str = Field(description="The exact short question to ask")
    follow_up_if_shallow: str = Field(default="", description="One follow-up if answer is vague")
    bridge_to_next: str = Field(default="", description="Conversational bridge to the next segment")

class InterviewPlan(BaseModel):
    opening_line: str = Field(description="Short welcome, role name, session framing")
    jd_summary_short: str = Field(description="2-3 sentence JD summary")
    strong_probed: str = Field(default="", description="Primary strength topic (score >= 8)")
    weak_probed: str = Field(default="", description="Weak/adjacent topic (score 5-7)")
    gap_probed: str = Field(default="", description="Gap topic (score < 5)")
    # Legacy aliases used by older cache entries
    strength_probed: str = Field(default="")
    segments: List[InterviewSegment] = Field(default_factory=list)
