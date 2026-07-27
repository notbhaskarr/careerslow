#!/usr/bin/env python3
"""
Sample candidate resume+JD pairs from HuggingFace for human labeling.

Outputs JSON candidates — NOT auto-labeled gold. Review and copy into dataset_realism.json.

Usage:
  pip install datasets
  python scripts/sample_realism_pairs.py --n 15 --out eval/golden/candidates.json
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def sample_pairs(dataset_name: str, n: int, seed: int = 42) -> list:
    """Sample diverse pairs from job_resume_fit-style dataset."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("Install datasets: pip install datasets") from e

    ds = load_dataset(dataset_name, split="train")
    by_category = defaultdict(list)
    for i, row in enumerate(ds):
        cat = row.get("category") or row.get("job_category") or "unknown"
        resume = row.get("resume_text") or row.get("resume") or ""
        job = row.get("job_text") or row.get("job") or row.get("jd") or ""
        if len(resume.split()) < 80 or len(job.split()) < 60:
            continue
        ai_score = row.get("ai_match_score")
        by_category[cat].append(
            {
                "index": i,
                "category": cat,
                "resume_word_count": len(resume.split()),
                "job_word_count": len(job.split()),
                "ai_match_score": ai_score,
                "resume_preview": resume[:300],
                "job_preview": job[:300],
                "resume_text": resume,
                "job_text": job,
                "job_required_skills": row.get("job_required_skills"),
                "resume_skill_list": row.get("resume_skill_list"),
            }
        )

    random.seed(seed)
    selected = []
    categories = sorted(by_category.keys())
    per_cat = max(1, n // max(len(categories), 1))

    for cat in categories:
        rows = by_category[cat]
        random.shuffle(rows)
        # Prefer mix of high and low ai_match_score when available
        rows.sort(key=lambda r: (r.get("ai_match_score") is None, r.get("ai_match_score") or 0))
        mid = len(rows) // 2
        picks = rows[: per_cat // 2] + rows[mid : mid + per_cat - per_cat // 2]
        selected.extend(picks[:per_cat])

    random.shuffle(selected)
    return selected[:n]


def main():
    parser = argparse.ArgumentParser(description="Sample realism eval candidates from HuggingFace")
    parser.add_argument("--dataset", default="batuhanmtl/job_resume_fit")
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="eval/golden/candidates.json")
    args = parser.parse_args()

    pairs = sample_pairs(args.dataset, args.n, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_dataset": args.dataset,
        "note": "Candidates for human labeling — not gold labels. See eval/golden/SAMPLING_PLAN.md",
        "count": len(pairs),
        "candidates": pairs,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {len(pairs)} candidates to {out}")
    cats = {p["category"] for p in pairs}
    print(f"Categories: {', '.join(sorted(cats))}")


if __name__ == "__main__":
    main()
