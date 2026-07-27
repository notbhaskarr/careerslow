"""
Evaluate Qdrant retrieval against golden `relevant_resume_snippets` labels.

Evaluates both skill queries (required/nice) and duty themes (responsibility).

Usage:
  python -m src.eval.run_retrieval_eval --dry-run
  python -m src.eval.run_retrieval_eval --layer control
  python -m src.eval.run_retrieval_eval --case backend_python
  python -m src.eval.run_retrieval_eval --output eval/reports/retrieval_latest.json
"""

import argparse
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

from src.db.qdrant_client import (
    DUTY_RETRIEVAL_SECTIONS,
    SKILL_RETRIEVAL_SECTIONS,
    VectorDatabase,
)
from src.eval.metrics import snippet_hit_at_k, snippet_mrr, snippet_precision_at_k
from src.eval.run_golden_eval import load_datasets, validate_dataset
from src.parser.document_parser import DocumentParser

logger = logging.getLogger(__name__)

REPORT_DIR = Path(__file__).resolve().parents[2] / "eval" / "reports"
SKILLS_BASELINE_PATH = REPORT_DIR / "retrieval_skills_baseline.json"


@dataclass
class RetrievalRequirementEval:
    requirement: str
    category: str  # "skill" | "duty"
    gold_bucket: str
    snippets: List[str]
    hit_at_3: bool = False
    precision_at_3: float = 0.0
    mrr: float = 0.0
    evidence_count: int = 0
    top_hits: List[dict] = field(default_factory=list)


@dataclass
class RetrievalCaseEval:
    case_id: str
    job_title: str
    layer: str
    resume_id: str
    requirements: List[RetrievalRequirementEval] = field(default_factory=list)
    error: Optional[str] = None

    def labeled_for(self, category: Optional[str] = None) -> List[RetrievalRequirementEval]:
        items = [r for r in self.requirements if r.snippets]
        if category:
            items = [r for r in items if r.category == category]
        return items

    @property
    def labeled_requirements(self) -> List[RetrievalRequirementEval]:
        return self.labeled_for()

    def hit_rate_at_3_for(self, category: Optional[str] = None) -> float:
        labeled = self.labeled_for(category)
        if not labeled:
            return 0.0
        return sum(1 for r in labeled if r.hit_at_3) / len(labeled)

    @property
    def hit_rate_at_3(self) -> float:
        return self.hit_rate_at_3_for()

    def mean_precision_at_3_for(self, category: Optional[str] = None) -> float:
        labeled = self.labeled_for(category)
        if not labeled:
            return 0.0
        return sum(r.precision_at_3 for r in labeled) / len(labeled)

    @property
    def mean_precision_at_3(self) -> float:
        return self.mean_precision_at_3_for()

    def mean_mrr_for(self, category: Optional[str] = None) -> float:
        labeled = self.labeled_for(category)
        if not labeled:
            return 0.0
        return sum(r.mrr for r in labeled) / len(labeled)

    @property
    def mean_mrr(self) -> float:
        return self.mean_mrr_for()


def _labeled_items(case: dict) -> List[tuple[str, dict]]:
    """Return (category, gold_row) pairs for skill + duty eval rows."""
    items: List[tuple[str, dict]] = []
    for row in case.get("requirements", []):
        items.append(("skill", row))
    for row in case.get("duty_requirements", []):
        items.append(("duty", row))
    return items


def evaluate_case_retrieval(
    case: dict,
    layer: str,
    parser: DocumentParser,
    db: VectorDatabase,
    limit: int = 3,
) -> RetrievalCaseEval:
    resume_id = f"reval_{case['id']}_{uuid.uuid4().hex[:6]}"
    result = RetrievalCaseEval(
        case_id=case["id"],
        job_title=case.get("job_title", ""),
        layer=layer,
        resume_id=resume_id,
    )
    try:
        parsed = parser.parse_resume(case["raw_resume"])
        parsed.raw_text = case["raw_resume"]
        db.index_resume(parsed, resume_id=resume_id)
    except Exception as e:
        result.error = f"index failed: {e}"
        return result

    for category, gold in _labeled_items(case):
        req = gold["requirement"]
        snippets = gold.get("relevant_resume_snippets") or []
        section_types = (
            DUTY_RETRIEVAL_SECTIONS if category == "duty" else SKILL_RETRIEVAL_SECTIONS
        )
        req_eval = RetrievalRequirementEval(
            requirement=req,
            category=category,
            gold_bucket=gold.get("gold_bucket", ""),
            snippets=snippets,
        )
        try:
            hits = db.retrieve_evidence_hits(
                query=req,
                resume_id=resume_id,
                limit=limit,
                section_types=section_types,
            )
        except Exception as e:
            result.error = f"retrieval failed for {req!r}: {e}"
            result.requirements.append(req_eval)
            return result

        texts = [h.text for h in hits]
        req_eval.evidence_count = len(texts)
        req_eval.top_hits = [
            {"text": h.text, "section_type": h.section_type, "score": round(h.score, 4)}
            for h in hits
        ]
        if snippets:
            req_eval.hit_at_3 = snippet_hit_at_k(texts, snippets, limit)
            req_eval.precision_at_3 = snippet_precision_at_k(texts, snippets, limit)
            req_eval.mrr = snippet_mrr(texts, snippets)
        result.requirements.append(req_eval)

    try:
        db.delete_resume_chunks(resume_id)
    except Exception:
        logger.warning("Failed to clean up eval chunks for %s", resume_id)
    return result


def _aggregate(cases: List[RetrievalCaseEval], category: Optional[str] = None) -> dict:
    labeled: List[RetrievalRequirementEval] = []
    for case in cases:
        labeled.extend(case.labeled_for(category))
    if not labeled:
        return {
            "cases": len(cases),
            "labeled_requirements": 0,
            "hit_rate_at_3": 0.0,
            "mean_precision_at_3": 0.0,
            "mean_mrr": 0.0,
        }
    return {
        "cases": len(cases),
        "labeled_requirements": len(labeled),
        "hit_rate_at_3": round(
            sum(1 for r in labeled if r.hit_at_3) / len(labeled), 4
        ),
        "mean_precision_at_3": round(
            sum(r.precision_at_3 for r in labeled) / len(labeled), 4
        ),
        "mean_mrr": round(sum(r.mrr for r in labeled) / len(labeled), 4),
    }


def _delta(current: float, baseline: float) -> float:
    return round(current - baseline, 4)


def _compare_to_baseline(current: dict, baseline_path: Path) -> Optional[dict]:
    if not baseline_path.exists():
        return None
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    base_agg = baseline.get("aggregate", {})
    skills_agg = current.get("aggregate_skills") or current.get("aggregate", {})
    comparison = {
        "baseline_file": str(baseline_path.name),
        "baseline_generated_at": baseline.get("generated_at"),
        "skills": {
            "baseline": base_agg,
            "current": skills_agg,
            "delta": {
                "hit_rate_at_3": _delta(
                    skills_agg.get("hit_rate_at_3", 0), base_agg.get("hit_rate_at_3", 0)
                ),
                "mean_precision_at_3": _delta(
                    skills_agg.get("mean_precision_at_3", 0),
                    base_agg.get("mean_precision_at_3", 0),
                ),
                "mean_mrr": _delta(
                    skills_agg.get("mean_mrr", 0), base_agg.get("mean_mrr", 0)
                ),
            },
        },
    }
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Golden-set retrieval evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Validate datasets only")
    parser.add_argument("--layer", default="control", choices=["control", "realism", "all"])
    parser.add_argument("--case", help="Run a single case id")
    parser.add_argument("--limit", type=int, default=3, help="Retrieval top-k")
    parser.add_argument("--output", type=Path, help="Write JSON report path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    datasets = load_datasets(args.layer)
    for layer_name, data in datasets:
        errors = validate_dataset(data, layer_name)
        if errors:
            raise SystemExit("\n".join(errors))

    if args.dry_run:
        total_cases = sum(len(d["cases"]) for _, d in datasets)
        duty_rows = sum(
            len(c.get("duty_requirements", []))
            for _, d in datasets
            for c in d["cases"]
        )
        print(
            f"Dry run OK — {total_cases} cases, {duty_rows} duty_requirements across layer(s) {args.layer}"
        )
        return

    doc_parser = DocumentParser()
    db = VectorDatabase()
    all_cases: List[RetrievalCaseEval] = []

    for layer_name, data in datasets:
        for case in data.get("cases", []):
            if args.case and case["id"] != args.case:
                continue
            logger.info("Evaluating retrieval: %s (%s)", case["id"], layer_name)
            all_cases.append(evaluate_case_retrieval(case, layer_name, doc_parser, db, args.limit))

    aggregate_skills = _aggregate(all_cases, category="skill")
    aggregate_duties = _aggregate(all_cases, category="duty")
    aggregate_all = _aggregate(all_cases)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "layer": args.layer,
        "limit": args.limit,
        "section_types": {
            "skill": SKILL_RETRIEVAL_SECTIONS,
            "duty": DUTY_RETRIEVAL_SECTIONS,
        },
        "aggregate": aggregate_all,
        "aggregate_skills": aggregate_skills,
        "aggregate_duties": aggregate_duties,
        "cases": [
            {
                **{k: v for k, v in asdict(c).items() if k != "requirements"},
                "hit_rate_at_3": round(c.hit_rate_at_3, 4),
                "mean_precision_at_3": round(c.mean_precision_at_3, 4),
                "mean_mrr": round(c.mean_mrr, 4),
                "skills": {
                    "hit_rate_at_3": round(c.hit_rate_at_3_for("skill"), 4),
                    "mean_precision_at_3": round(c.mean_precision_at_3_for("skill"), 4),
                    "mean_mrr": round(c.mean_mrr_for("skill"), 4),
                },
                "duties": {
                    "hit_rate_at_3": round(c.hit_rate_at_3_for("duty"), 4),
                    "mean_precision_at_3": round(c.mean_precision_at_3_for("duty"), 4),
                    "mean_mrr": round(c.mean_mrr_for("duty"), 4),
                },
                "requirements": [asdict(r) for r in c.requirements],
            }
            for c in all_cases
        ],
    }

    comparison = _compare_to_baseline(report, SKILLS_BASELINE_PATH)
    if comparison:
        report["comparison_to_skills_baseline"] = comparison

    out = args.output or REPORT_DIR / "retrieval_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "aggregate_all": aggregate_all,
            "aggregate_skills": aggregate_skills,
            "aggregate_duties": aggregate_duties,
            "comparison_to_skills_baseline": comparison,
        },
        indent=2,
    ))
    print(f"Report → {out}")


if __name__ == "__main__":
    main()
