"""Deterministic cache key helpers for resume/JD pairs and plan versioning."""

import hashlib
import os

# Bump when interview plan selection logic changes (invalidates cached plans).
PLAN_VERSION = os.getenv("INTERVIEW_PLAN_VERSION", "2")


def hash_resume_text(text: str) -> str:
    """Return MD5 hex digest of raw resume text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def hash_jd_text(text: str) -> str:
    """Return MD5 hex digest of normalized JD text (whitespace-collapsed, lowercased)."""
    normalized = " ".join(text.split()).strip().lower()
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def make_resume_id(resume_text: str) -> str:
    """Public resume identifier derived from content hash."""
    return f"res_{hash_resume_text(resume_text)}"


def make_jd_id(jd_text: str) -> str:
    """Public JD identifier derived from normalized content hash."""
    return f"jd_{hash_jd_text(jd_text)}"


def make_pair_id(resume_text: str, jd_text: str) -> str:
    """
    Composite key for a resume+JD analysis session.

    Same resume against two JDs yields two pair_ids (no cache collision).
    """
    return f"pair_{hash_resume_text(resume_text)}_{hash_jd_text(jd_text)}"


def plan_cache_key(pair_id: str) -> str:
    """Redis suffix for interview plans including plan logic version."""
    return f"{pair_id}:v{PLAN_VERSION}"
