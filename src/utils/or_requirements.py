"""Parse JD OR-group requirements and count explicit option matches in evidence."""

import re
from typing import List

_OR_PREFIX = re.compile(r"^at least one of:\s*", re.IGNORECASE)
_SPLIT = re.compile(r",|\s+or\s+", re.IGNORECASE)
_SKIP = frozenset({"etc", "etcetera", "similar", "similar tools"})


def parse_or_options(requirement: str) -> List[str]:
    """Return listed OR options, or empty if not an OR-group requirement."""
    text = (requirement or "").strip()
    if not _OR_PREFIX.match(text):
        return []
    body = _OR_PREFIX.sub("", text, count=1).strip()
    options: List[str] = []
    for part in _SPLIT.split(body):
        token = part.strip().strip(".")
        if not token:
            continue
        if token.lower() in _SKIP or token.lower().startswith("similar"):
            continue
        options.append(token)
    return options


def _term_in_text(term: str, blob: str) -> bool:
    escaped = re.escape(term.lower())
    pattern = rf"(?<![a-zA-Z0-9#]){escaped}(?![a-zA-Z0-9#])"
    return bool(re.search(pattern, blob, re.IGNORECASE))


def count_or_matches(options: List[str], evidence: List[str]) -> List[str]:
    """Return which OR options appear explicitly in the evidence snippets."""
    if not options:
        return []
    blob = " ".join(evidence)
    return [opt for opt in options if _term_in_text(opt, blob)]


def apply_or_match_score(matched_count: int, llm_score: int) -> int:
    """
    Enforce OR-group scoring: 2+ listed matches → 10, 1 → 9, 0 → cap at 7.
    """
    if matched_count >= 2:
        return 10
    if matched_count == 1:
        return 9
    return min(llm_score, 7)
