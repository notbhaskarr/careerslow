import os
from langchain_google_genai import ChatGoogleGenerativeAI
from src.schemas.document_schemas import ParsedResume, ParsedJD

class DocumentParser:
    def __init__(self):
        """
        Initializes the DocumentParser using the Gemini LLM for structured extraction.
        
        Inputs:
        - None
        
        Returns:
        - None
        """
        # Resume: temp=0 for deterministic extraction; JD: temp=0.1 for slight flexibility on OR-group parsing
        self.resume_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
        self.jd_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.1)

        # LangChain's magic: forces the LLM to output exactly our Pydantic schemas!
        self.resume_extractor = self.resume_llm.with_structured_output(ParsedResume)
        self.jd_extractor = self.jd_llm.with_structured_output(ParsedJD)

    def parse_resume(self, text: str) -> ParsedResume:
        """
        Parses a raw resume string into structured data using the Gemini LLM.
        
        Inputs:
        - text (str): The raw, unstructured text of a candidate's resume.
        
        Returns:
        - ParsedResume: A Pydantic object containing the categorized skills, experience, projects, and domains.
        """
        prompt = (
            "You are an expert technical recruiter parsing a resume for automated job matching.\n"
            "Each list item you output becomes one search chunk in a vector database — optimize for retrieval quality.\n\n"
            "OUTPUT FIELDS:\n"
            "- technical_skills: tools, languages, frameworks, platforms, testing methods, certifications.\n"
            "- soft_skills: interpersonal traits only (leadership, communication, collaboration).\n"
            "- experience_sections: complete work-experience bullets as standalone sentences.\n"
            "- projects: complete project descriptions as standalone sentences.\n"
            "- domain_expertise: industries, business domains, verticals explicitly stated.\n\n"
            "TECHNICAL_SKILLS (each entry = one search chunk):\n"
            "1. Use ONLY terms explicitly stated in the resume — never invent or import skills from examples.\n"
            "2. Preserve compound names exactly as written (e.g. grouped slash names, multi-word tool names).\n"
            "3. When the resume lists multiple skills on one line or under one heading, keep them as ONE entry "
            "(e.g. 'LangA, LangB, LangC' → one entry 'LangA, LangB, LangC', not three separate entries).\n"
            "4. Do NOT emit bare ambiguous tokens (single letters, 1-2 char abbreviations) as standalone entries. "
            "Use the most specific form the resume provides (e.g. 'Core LangB' not bare 'B').\n"
            "5. Methodologies and process frameworks (Agile, Scrum, Kanban) → technical_skills, NOT soft_skills.\n\n"
            "EXPERIENCE & PROJECTS:\n"
            "- Extract as complete, standalone sentences — not fragments or bullet prefixes alone.\n"
            "- Keep named tools and technologies in context within the sentence (aids retrieval).\n"
            "- One bullet = one entry. Include outcomes/metrics when present in the resume.\n\n"
            "SOFT_SKILLS:\n"
            "- Interpersonal and leadership traits only. Skip generic filler with no resume evidence.\n"
            "- Do NOT put tools, languages, or methodologies here.\n\n"
            "DOMAIN_EXPERTISE:\n"
            "- Industries and business domains only. Do not duplicate items already in technical_skills.\n\n"
            "CRITICAL: Do NOT infer or hallucinate. If a section is absent, leave the list empty.\n\n"
            "ANTI-PATTERNS (wrong):\n"
            "- Splitting 'LangA, LangB, LangC' into three separate technical_skills entries.\n"
            "- Standalone single-letter skill entry that could match unrelated searches.\n"
            "- One-word experience fragment with no context.\n"
            "- Agile/Scrum listed under soft_skills.\n"
            "- Skills copied from these examples that do not appear in the resume.\n\n"
            f"Resume Text:\n{text}"
        )
        # The LLM reads the prompt and returns a fully populated ParsedResume object
        return self.resume_extractor.invoke(prompt)

    def parse_jd(self, text: str) -> ParsedJD:
        """
        Parses a raw JD string into structured requirements using the Gemini LLM.
        
        Inputs:
        - text (str): The raw, unstructured text of a Job Description.
        
        Returns:
        - ParsedJD: A Pydantic object containing required skills, nice-to-haves, responsibilities, and qualifications.
        """
        prompt = (
            "You are an expert technical recruiter preparing requirements for automated resume matching.\n"
            "Each skill_name becomes a search query against candidate resumes — optimize for retrieval.\n\n"
            "OUTPUT FIELDS:\n"
            "- required_skills / nice_to_have_skills: scorable skill/tool requirements. Target 8-10 combined (min 5, max 12).\n"
            "- responsibility_themes: 5-8 scorable duty themes from Principal Accountabilities / job duties (see below).\n"
            "- responsibilities: full duty sentences for interview context (NOT individually scored).\n"
            "- qualifications: degrees, years of experience, certifications as complete sentences.\n"
            "- domain_or_industry / job_title: as stated in the JD.\n\n"
            "SCAN THE FULL JD:\n"
            "- Read every section regardless of header ('Requirements', 'Qualifications', 'What you'll do', etc.).\n"
            "- Classify by language strength ('must', 'required', 'minimum' vs 'preferred', 'plus', 'bonus'), not section title alone.\n"
            "- Lift named tools/skills from qualifications into required_skills when the JD ties them to experience years or must-haves.\n\n"
            "REQUIREMENTS vs RESPONSIBILITIES:\n"
            "- required_skills / nice_to_have_skills: named tools, languages, frameworks, methods (Knowledge/Skills/Abilities block).\n"
            "- responsibility_themes: summarized DUTY areas from accountabilities / 'what you'll do' — searchable labels for experience matching.\n"
            "- responsibilities: longer verbatim duty sentences (for interview plans); do NOT score individually.\n"
            "- Do NOT duplicate: if a duty is covered by a responsibility_theme, do not also add it as a required_skill unless it names a specific tool.\n\n"
            "RESPONSIBILITY THEMES (scored for experience alignment):\n"
            "- Extract 5-8 themes from job duty / accountabilities sections (e.g. Quality Thinking 25%, Test Planning 50%).\n"
            "- theme_name: search-optimized label using exact JD terms, under ~12 words "
            "(e.g. 'Exploratory regression E2E testing', 'LLM output validation and prompt regression').\n"
            "- importance_weight: 1-9 max. Use JD time-allocation when stated (50% block → 9; 25% → 7-8); else 8.\n"
            "- Merge related duty bullets into one theme; do not emit 20 separate themes.\n"
            "- Use placeholder-style grouping: risk/acceptance criteria review, test strategy execution, defect management, AI-assisted testing, scrum collaboration.\n\n"
            "HARD vs SOFT SKILLS:\n"
            "- Hard skills (tools, languages, frameworks, methods): always emit when explicitly stated.\n"
            "- Soft skills ('communication', 'problem-solving', 'team player'): SKIP unless the JD gives searchable anchors "
            "(e.g. 'document test cases', 'present findings to stakeholders'). Generic boilerplate → omit from skill lists.\n\n"
            "SKILL_NAME RULES (each skill_name is a search query):\n"
            "1. Use ONLY terms explicitly stated in the Job Description Text — never invent or import skills from examples.\n"
            "2. Lead with the primary keyword from the JD. Keep under ~12 words.\n"
            "3. Strip filler: 'Experience with', 'Exposure to', 'Familiarity with'.\n"
            "4. Use exact JD spelling for easily confused terms (keep the most specific form the JD uses).\n"
            "5. One evaluable unit per entry. Merge overlaps; do not duplicate across OR groups and separate entries.\n\n"
            "OR vs AND:\n"
            "- OR (candidate needs ONE): 'or', 'and/or', comma + trailing 'or [item]', slashes (ToolA/ToolB), "
            "pipes (ToolA|ToolB), or parenthetical alternatives (ToolA, ToolB).\n"
            "  → One entry: skill_name='At least one of: [exact terms from JD]'.\n"
            "  → Do NOT split into separate entries per option.\n"
            "- AND (candidate needs ALL): '[X] and [Y]', 'both [X] and [Y]', clearly conjunctive.\n"
            "  → Separate entries, one per skill, using exact JD terms.\n\n"
            "CLASSIFICATION & WEIGHTS:\n"
            "- 'Required' / 'must have' / 'minimum' → required_skills: weight 7-9.\n"
            "- Main requirements block, no qualifier → required_skills: weight 8.\n"
            "- 'Preferred' / 'plus' / 'bonus' / 'nice-to-have' → nice_to_have_skills: weight 5-6.\n"
            "- Mentioned once in passing → nice_to_have_skills: weight 3-4.\n"
            "- Ambiguous → default nice_to_have_skills.\n\n"
            "SKIP (never emit as skill requirements):\n"
            "- EEO/diversity, benefits, salary, location, visa sponsorship.\n"
            "- Generic soft-skill boilerplate without searchable JD anchors.\n"
            "- Duplicate or overlapping entries already covered by an OR group.\n\n"
            "VOLUME:\n"
            "- Extract the minimum distinct set needed to capture the JD. Prefer fewer, sharper requirements over exhaustive lists.\n"
            "- If the JD lists many similar tools as alternatives, use one OR-group entry, not one entry per tool.\n\n"
            "CRITICAL: Do NOT infer or hallucinate. Extract only what is explicitly stated in the JD text.\n\n"
            "STRUCTURAL EXAMPLES (format only — never output skills that appear only here):\n"
            "JD phrase: 'Must know Alpha and Beta. Gamma or Delta preferred.'\n"
            "→ required_skills: [{skill_name: 'Alpha', weight: 9}, {skill_name: 'Beta', weight: 9}]\n"
            "→ nice_to_have_skills: [{skill_name: 'At least one of: Gamma, Delta', weight: 5}]\n\n"
            "JD phrase: 'ToolOne/ToolTwo/ToolThree for [task in JD].'\n"
            "→ required_skills: [{skill_name: 'At least one of: ToolOne, ToolTwo, ToolThree', weight: 8}]\n\n"
            "ANTI-PATTERNS (wrong):\n"
            "- 'LangA or LangB' → two separate entries LangA and LangB.\n"
            "- OR group already covers LangA → also emit separate 'Experience with LangA'.\n"
            "- skill_name: 'Experience with exposure to ToolX' (filler, not keyword-first).\n"
            "- 15 generic soft-skill requirements with no searchable terms.\n"
            "- Skills copied from these examples that do not appear in the JD.\n\n"
            f"Job Description Text:\n{text}"
        )
        # The LLM reads the prompt and returns a fully populated ParsedJD object
        return self.jd_extractor.invoke(prompt)
