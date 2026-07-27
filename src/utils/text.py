import re


def resume_excerpt(raw: str, limit: int = 1500) -> str:
    text = " ".join(raw.split())
    return text[:limit] + ("..." if len(text) > limit else "")


def extract_candidate_name(raw_resume: str) -> str:
    """Best-effort first name from the resume header line."""
    for line in raw_resume.splitlines():
        cleaned = line.strip()
        if not cleaned or len(cleaned) > 80:
            continue
        if re.search(r"@|https?://|linkedin|github|\d{3}[-.\s]?\d{3}", cleaned, re.I):
            continue
        words = cleaned.split()
        if 1 <= len(words) <= 5 and re.match(r"^[A-Za-z][A-Za-z\-'.]*$", words[0]):
            return words[0]
    return ""
