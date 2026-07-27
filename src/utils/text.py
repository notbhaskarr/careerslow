def resume_excerpt(raw: str, limit: int = 1500) -> str:
    text = " ".join(raw.split())
    return text[:limit] + ("..." if len(text) > limit else "")
