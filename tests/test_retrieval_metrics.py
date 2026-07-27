"""Tests for snippet-based retrieval metrics."""

from src.eval.metrics import (
    snippet_hit_at_k,
    snippet_mrr,
    snippet_precision_at_k,
    text_contains_snippet,
)


def test_text_contains_snippet_case_insensitive():
    assert text_contains_snippet("Built FastAPI microservices", "fastapi")
    assert not text_contains_snippet("PostgreSQL production use", "mysql")


def test_snippet_hit_at_k():
    retrieved = [
        "Deployed with Docker Compose",
        "Built FastAPI microservices with Redis",
        "React dashboard",
    ]
    snippets = ["FastAPI", "Redis"]
    assert snippet_hit_at_k(retrieved, snippets, 3) is True
    assert snippet_hit_at_k(retrieved, snippets, 1) is False


def test_snippet_precision_at_k():
    retrieved = ["Python and FastAPI", "React only", "Redis caching"]
    snippets = ["FastAPI", "Redis"]
    assert round(snippet_precision_at_k(retrieved, snippets, 3), 2) == round(2 / 3, 2)


def test_snippet_mrr():
    retrieved = ["React", "PostgreSQL production use", "JavaScript"]
    snippets = ["PostgreSQL"]
    assert snippet_mrr(retrieved, snippets) == 0.5


def test_empty_snippets_not_applicable():
    assert snippet_hit_at_k(["anything"], [], 3) is False
    assert snippet_mrr(["anything"], []) == 0.0
