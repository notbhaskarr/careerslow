"""Apply accepted resume suggestions to plain-text resume drafts."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List, Optional

from pydantic import BaseModel, Field


class AcceptedEdit(BaseModel):
    action: str = Field(description="'edit' or 'add'")
    target_line: str = Field(default="")
    text: str = Field(description="Revised or new line text")


def _normalize(s: str) -> str:
    return " ".join(s.split()).strip().lower()


def _replace_once(haystack: str, needle: str, replacement: str) -> Optional[str]:
    if not needle.strip():
        return None
    if needle in haystack:
        return haystack.replace(needle, replacement, 1)
    # Line-by-line normalized match
    lines = haystack.splitlines()
    target_norm = _normalize(needle)
    for i, line in enumerate(lines):
        if _normalize(line) == target_norm or target_norm in _normalize(line):
            lines[i] = replacement
            return "\n".join(lines)
    return None


def _fuzzy_replace(haystack: str, needle: str, replacement: str, threshold: float = 0.55) -> str:
    """Replace the best-matching line when exact match fails."""
    if not needle.strip():
        return haystack
    lines = haystack.splitlines()
    target_norm = _normalize(needle)
    best_idx = -1
    best_ratio = 0.0
    for i, line in enumerate(lines):
        line_norm = _normalize(line)
        if not line_norm:
            continue
        ratio = SequenceMatcher(None, target_norm, line_norm).ratio()
        if target_norm in line_norm or line_norm in target_norm:
            ratio = max(ratio, 0.85)
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i
    if best_idx >= 0 and best_ratio >= threshold:
        lines[best_idx] = replacement
        return "\n".join(lines)
    return haystack


def apply_edits(resume_text: str, edits: List[AcceptedEdit]) -> str:
    """Merge accepted edits into resume text; append adds at the end."""
    text = resume_text
    adds: List[str] = []

    for edit in edits:
        action = (edit.action or "").lower()
        if action == "edit":
            updated = _replace_once(text, edit.target_line, edit.text)
            if updated is not None:
                text = updated
            else:
                text = _fuzzy_replace(text, edit.target_line, edit.text)
        elif action == "add" and edit.text.strip():
            adds.append(edit.text.strip())

    if adds:
        block = "\n".join(f"• {line.lstrip('•').strip()}" for line in adds)
        text = text.rstrip() + "\n\n" + block + "\n"
    return text
