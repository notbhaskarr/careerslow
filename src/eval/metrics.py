"""
Offline evaluation metrics for CareersLow AI pipelines.

Retrieval (Qdrant): precision@k, recall@k, MRR — require labeled golden chunks.
Gap scoring: bucket accuracy / MAE vs human labels.
Debrief: topic grounding rate, study_topics non-empty rate.

Run golden-set evals via LangSmith (see src/eval/evaluate.py).
"""

from typing import List, Set


def _normalize(text: str) -> str:
    return text.lower().strip()


def text_contains_snippet(text: str, snippet: str) -> bool:
    """Case-insensitive substring match for golden retrieval labels."""
    snippet = _normalize(snippet)
    if not snippet:
        return False
    return snippet in _normalize(text)


def snippet_hit_at_k(retrieved_texts: List[str], snippets: List[str], k: int) -> bool:
    """
    True if any top-k retrieved chunk contains at least one labeled snippet.
    Empty snippets → not applicable (returns False).
    """
    if k <= 0 or not snippets:
        return False
    for text in retrieved_texts[:k]:
        if any(text_contains_snippet(text, snippet) for snippet in snippets):
            return True
    return False


def snippet_precision_at_k(retrieved_texts: List[str], snippets: List[str], k: int) -> float:
    """
    Fraction of top-k retrieved chunks that contain at least one gold snippet.
    """
    if k <= 0 or not snippets:
        return 0.0
    top = retrieved_texts[:k]
    if not top:
        return 0.0
    hits = sum(
        1
        for text in top
        if any(text_contains_snippet(text, snippet) for snippet in snippets)
    )
    return hits / len(top)


def snippet_mrr(retrieved_texts: List[str], snippets: List[str]) -> float:
    """Reciprocal rank of the first retrieved chunk matching any gold snippet."""
    if not snippets:
        return 0.0
    for i, text in enumerate(retrieved_texts):
        if any(text_contains_snippet(text, snippet) for snippet in snippets):
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Fraction of top-k retrieved items that are relevant.

    Args:
        retrieved_ids: Ranked retrieved chunk IDs (best first).
        relevant_ids: Ground-truth relevant chunk IDs for the query.
        k: Cutoff rank.

    Returns:
        Precision@k in [0, 1], or 0.0 if k is 0.
    """
    if k <= 0:
        return 0.0
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for cid in top if cid in relevant_ids)
    return hits / len(top)


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Fraction of all relevant items found in top-k.

    Returns:
        Recall@k in [0, 1], or 0.0 if no relevant items exist.
    """
    if not relevant_ids or k <= 0:
        return 0.0
    top = set(retrieved_ids[:k])
    hits = len(top & relevant_ids)
    return hits / len(relevant_ids)


def mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """Mean reciprocal rank of the first relevant hit (1/rank), or 0 if none."""
    for i, cid in enumerate(retrieved_ids):
        if cid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def score_bucket(match_score: int) -> str:
    """Map 1-10 match score to strong / weak / gap bucket."""
    if match_score >= 8:
        return "strong"
    if match_score >= 5:
        return "weak"
    return "gap"


def bucket_accuracy(predicted_scores: List[int], gold_buckets: List[str]) -> float:
    """
    Accuracy of score bucket classification vs human labels.

    Args:
        predicted_scores: Model match_score per requirement.
        gold_buckets: Human-labeled bucket names (strong/weak/gap).

    Returns:
        Fraction of exact bucket matches.
    """
    if not predicted_scores or len(predicted_scores) != len(gold_buckets):
        return 0.0
    hits = sum(
        1 for s, g in zip(predicted_scores, gold_buckets) if score_bucket(s) == g
    )
    return hits / len(gold_buckets)
