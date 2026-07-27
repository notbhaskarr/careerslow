"""Tests for offline eval metrics."""

from src.eval.metrics import precision_at_k, recall_at_k, mrr, score_bucket, bucket_accuracy


def test_precision_at_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c"}
    assert precision_at_k(retrieved, relevant, 2) == 0.5


def test_recall_at_k():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "c", "e"}
    assert recall_at_k(retrieved, relevant, 3) == 2 / 3


def test_mrr_first_rank():
    assert mrr(["x", "a", "b"], {"a"}) == 0.5


def test_score_bucket():
    assert score_bucket(9) == "strong"
    assert score_bucket(6) == "weak"
    assert score_bucket(3) == "gap"


def test_bucket_accuracy():
    assert bucket_accuracy([9, 6, 3], ["strong", "weak", "gap"]) == 1.0
