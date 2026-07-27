"""
Debug helpers for resume extraction (PDF) and LLM parsing.

Enable full text/file dumps with DEBUG_RESUME_PIPELINE=true (default: true for local QA).
Logs always include counts and short previews (no full resume in INFO unless debug on).
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEBUG_DIR = Path(__file__).resolve().parents[2] / "eval" / "debug_runs"
PREVIEW_CHARS = 300


def is_pipeline_debug_enabled() -> bool:
    """Whether to write full debug files and verbose log previews."""
    return os.getenv("DEBUG_RESUME_PIPELINE", "true").lower() in ("1", "true", "yes")


def parsed_resume_summary(parsed: Any) -> Dict[str, Any]:
    """Build a structured summary of ParsedResume list fields for logs and API."""
    fields = (
        "technical_skills",
        "soft_skills",
        "experience_sections",
        "projects",
        "domain_expertise",
    )
    sections = {}
    total_items = 0
    for name in fields:
        items = getattr(parsed, name, None) or []
        sections[name] = {"count": len(items), "items": list(items)}
        total_items += len(items)
    raw_len = len(getattr(parsed, "raw_text", "") or "")
    return {
        "raw_text_chars": raw_len,
        "total_list_items": total_items,
        "sections": sections,
    }


def log_extraction(
    resume_id: str,
    filename: str,
    page_texts: List[str],
    resume_text: str,
) -> Dict[str, Any]:
    """
    Log PDF extraction stats; optionally write eval/debug_runs/{resume_id}_extracted.txt.

    Returns:
        Dict suitable for Redis extract:{resume_id} and debug API.
    """
    page_char_counts = [len(p) for p in page_texts]
    payload = {
        "resume_id": resume_id,
        "source_filename": filename,
        "page_count": len(page_texts),
        "page_char_counts": page_char_counts,
        "char_count": len(resume_text),
        "text": resume_text,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }

    preview = resume_text[:PREVIEW_CHARS].replace("\n", " ")
    logger.info(
        "[EXTRACT] resume_id=%s file=%s pages=%s chars=%s preview=%r",
        resume_id,
        filename,
        len(page_texts),
        len(resume_text),
        preview,
    )
    print(
        f"\n{'='*60}\n[PIPELINE DEBUG — EXTRACTION]\n"
        f"  resume_id: {resume_id}\n"
        f"  file: {filename}\n"
        f"  pages: {len(page_texts)} | total chars: {len(resume_text)}\n"
        f"  per-page chars: {page_char_counts}\n"
        f"  preview: {preview}...\n{'='*60}\n"
    )

    if is_pipeline_debug_enabled():
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        out = DEBUG_DIR / f"{resume_id}_extracted.txt"
        header = (
            f"# resume_id: {resume_id}\n# file: {filename}\n"
            f"# pages: {len(page_texts)} | chars: {len(resume_text)}\n"
            f"# per-page chars: {page_char_counts}\n\n"
        )
        out.write_text(header + resume_text, encoding="utf-8")
        logger.info("[EXTRACT] wrote %s", out)
        print(f"[PIPELINE DEBUG] Full extraction text → {out}")

    return payload


def log_parsed_resume(
    resume_id: str,
    parsed: Any,
    *,
    from_cache: bool = False,
) -> Dict[str, Any]:
    """
    Log structured parse output; optionally write eval/debug_runs/{resume_id}_parsed.json.

    Returns:
        Summary dict for debug API.
    """
    summary = parsed_resume_summary(parsed)
    summary["resume_id"] = resume_id
    summary["from_cache"] = from_cache
    summary["parsed_at"] = datetime.now(timezone.utc).isoformat()

    print(f"\n{'='*60}\n[PIPELINE DEBUG — PARSE]{' (cache hit)' if from_cache else ''}\n")
    print(f"  resume_id: {resume_id}")
    print(f"  raw_text_chars: {summary['raw_text_chars']}")
    print(f"  total_list_items (future chunks): {summary['total_list_items']}")
    for name, block in summary["sections"].items():
        print(f"  {name}: {block['count']} items")
        for i, item in enumerate(block["items"][:3]):
            snippet = item[:120] + ("..." if len(item) > 120 else "")
            print(f"    [{i+1}] {snippet}")
        if block["count"] > 3:
            print(f"    ... +{block['count'] - 3} more")
    print(f"{'='*60}\n")

    logger.info(
        "[PARSE] resume_id=%s cache=%s total_items=%s sections=%s",
        resume_id,
        from_cache,
        summary["total_list_items"],
        {k: v["count"] for k, v in summary["sections"].items()},
    )

    if is_pipeline_debug_enabled() and not from_cache:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        out = DEBUG_DIR / f"{resume_id}_parsed.json"
        data = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("[PARSE] wrote %s", out)
        print(f"[PIPELINE DEBUG] Full parsed JSON → {out}")

        summary_path = DEBUG_DIR / f"{resume_id}_pipeline_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return summary


def log_chunks(
    resume_id: str,
    chunks: List[Dict[str, Any]],
    *,
    from_existing_index: bool = False,
) -> Dict[str, Any]:
    """
    Log Qdrant chunk payloads (section_type + text); write eval/debug_runs/{resume_id}_chunks.json.

    Args:
        resume_id: Resume identifier.
        chunks: List of dicts with at least section_type and text (optional chunk_index).
        from_existing_index: True when loaded from Qdrant without re-indexing.

    Returns:
        Summary dict for Redis and debug API.
    """
    by_section: Dict[str, int] = {}
    for c in chunks:
        st = c.get("section_type", "unknown")
        by_section[st] = by_section.get(st, 0) + 1

    payload = {
        "resume_id": resume_id,
        "chunk_count": len(chunks),
        "by_section_type": by_section,
        "from_existing_index": from_existing_index,
        "chunked_at": datetime.now(timezone.utc).isoformat(),
        "chunks": [
            {
                "index": i + 1,
                "section_type": c.get("section_type", "unknown"),
                "text": c.get("text", ""),
                "char_count": len(c.get("text", "") or ""),
            }
            for i, c in enumerate(chunks)
        ],
    }

    label = " (existing Qdrant index)" if from_existing_index else ""
    print(f"\n{'='*60}\n[PIPELINE DEBUG — CHUNKS]{label}\n")
    print(f"  resume_id: {resume_id}")
    print(f"  total chunks: {len(chunks)}")
    print(f"  by section_type: {by_section}")
    for i, c in enumerate(chunks[:5]):
        st = c.get("section_type", "?")
        text = c.get("text", "")
        snippet = text[:100] + ("..." if len(text) > 100 else "")
        print(f"  [{i+1}] ({st}) {snippet}")
    if len(chunks) > 5:
        print(f"  ... +{len(chunks) - 5} more chunks")
    print(f"{'='*60}\n")

    logger.info(
        "[CHUNKS] resume_id=%s count=%s existing=%s sections=%s",
        resume_id,
        len(chunks),
        from_existing_index,
        by_section,
    )

    if is_pipeline_debug_enabled():
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        out = DEBUG_DIR / f"{resume_id}_chunks.json"
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("[CHUNKS] wrote %s", out)
        print(f"[PIPELINE DEBUG] Full chunks JSON → {out}")

    return payload
