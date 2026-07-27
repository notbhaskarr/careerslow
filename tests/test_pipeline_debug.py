"""Tests for pipeline debug helpers."""

from src.schemas.document_schemas import ParsedResume
from src.utils.pipeline_debug import log_chunks, parsed_resume_summary


def test_log_chunks_summary():
    chunks = [
        {"section_type": "technical_skills", "text": "Python"},
        {"section_type": "experience", "text": "Built APIs"},
    ]
    payload = log_chunks("res_test", chunks, from_existing_index=False)
    assert payload["chunk_count"] == 2
    assert payload["by_section_type"]["technical_skills"] == 1


def test_parsed_resume_summary():
    parsed = ParsedResume(
        raw_text="Hello world",
        technical_skills=["Python", "FastAPI"],
        experience_sections=["Built APIs"],
        soft_skills=[],
        projects=[],
        domain_expertise=[],
    )
    summary = parsed_resume_summary(parsed)
    assert summary["total_list_items"] == 3
    assert summary["sections"]["technical_skills"]["count"] == 2
    assert summary["sections"]["experience_sections"]["items"][0] == "Built APIs"
