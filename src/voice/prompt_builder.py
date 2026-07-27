"""Builds the static system prompt for the voice session (no per-turn plan XML)."""

PROMPT_PATH = "voiceprompt.txt"


def build_system_prompt(job_title: str, jd_summary: str, resume_excerpt: str) -> str:
    with open(PROMPT_PATH, "r") as f:
        prompt = f.read()
    return (
        prompt.replace("[INJECT: Job Title]", job_title)
        .replace("[INJECT: JD Summary Short]", jd_summary)
        .replace("[INJECT: Resume Excerpt]", resume_excerpt)
    )
