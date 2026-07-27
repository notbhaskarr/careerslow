"""
Run golden-set evaluation against local datasets (eval/golden/).

Usage:
  python -m src.eval.run_golden_eval --dry-run
  python -m src.eval.run_golden_eval --layer all
  python -m src.eval.run_golden_eval --layer control
  python -m src.eval.run_golden_eval --layer realism
  python -m src.eval.run_golden_eval --case backend_python
  python -m src.eval.run_golden_eval --output eval/reports/latest.json
"""

import argparse
import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

from src.eval.metrics import bucket_accuracy, score_bucket
from src.graph.state import GraphState
from src.graph.workflow import WorkflowBuilder

logger = logging.getLogger(__name__)

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "eval" / "golden"
CONTROL_PATH = GOLDEN_DIR / "dataset.json"
REALISM_PATH = GOLDEN_DIR / "dataset_realism.json"

LAYER_PATHS = {
    "control": CONTROL_PATH,
    "realism": REALISM_PATH,
}


@dataclass
class RequirementEval:
    requirement: str
    gold_bucket: str
    predicted_score: Optional[int] = None
    predicted_bucket: Optional[str] = None
    in_score_range: bool = False
    bucket_match: bool = False


@dataclass
class CaseEval:
    case_id: str
    job_title: str
    layer: str = "control"
    requirements: List[RequirementEval] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def bucket_accuracy(self) -> float:
        if not self.requirements:
            return 0.0
        hits = sum(1 for r in self.requirements if r.bucket_match)
        return hits / len(self.requirements)


def load_golden_dataset(path: Path) -> dict:
    """Load and return parsed golden dataset JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_datasets(layer: str = "all") -> List[Tuple[str, dict]]:
    """
    Load one or both dataset files.

    Returns:
        List of (layer_name, dataset_dict) tuples.
    """
    if layer == "all":
        paths = [("control", CONTROL_PATH), ("realism", REALISM_PATH)]
    elif layer in LAYER_PATHS:
        paths = [(layer, LAYER_PATHS[layer])]
    else:
        raise ValueError(f"Unknown layer: {layer}. Use control, realism, or all.")

    loaded = []
    for name, path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        loaded.append((name, load_golden_dataset(path)))
    return loaded


def validate_dataset(data: dict, layer_name: str = "") -> List[str]:
    """
    Validate dataset structure without calling LLM.

    Returns:
        List of error strings; empty if valid.
    """
    errors: List[str] = []
    prefix = f"{layer_name}: " if layer_name else ""
    cases = data.get("cases", [])
    if not cases:
        errors.append(f"{prefix}No cases in dataset")
        return errors
    for case in cases:
        cid = case.get("id", "?")
        for req in case.get("requirements", []) + case.get("duty_requirements", []):
            bucket = req.get("gold_bucket")
            if bucket not in ("strong", "weak", "gap"):
                errors.append(f"{prefix}{cid}: invalid gold_bucket {bucket!r}")
            lo, hi = req.get("gold_score_min"), req.get("gold_score_max")
            if lo is not None and hi is not None and lo > hi:
                errors.append(
                    f"{prefix}{cid}: gold_score_min > gold_score_max for {req.get('requirement')}"
                )
    return errors


def _match_requirement(predicted_req: str, gold_req: str) -> bool:
    """Fuzzy match requirement strings from pipeline vs golden labels."""
    a = predicted_req.lower().strip()
    b = gold_req.lower().strip()
    if a == b or a in b or b in a:
        return True
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    overlap = len(a_tokens & b_tokens)
    return overlap >= max(2, min(len(a_tokens), len(b_tokens)) // 2)


async def evaluate_case(case: dict, builder: WorkflowBuilder, layer: str) -> CaseEval:
    """
    Run LangGraph pipeline for one golden case and score labeled requirements.
    """
    result = CaseEval(
        case_id=case["id"],
        job_title=case.get("job_title", ""),
        layer=case.get("layer", layer),
    )
    graph = builder.build_graph()
    state = GraphState(
        resume_id=f"eval_{case['id']}_{uuid.uuid4().hex[:6]}",
        raw_resume=case["raw_resume"],
        raw_jd=case["raw_jd"],
        parsed_resume=None,
        parsed_jd=None,
        gap_analyses=[],
        overall_fit_score=0.0,
        score_breakdown={},
        errors=[],
        skip_indexing=False,
        retry_count=0,
        is_grounded=True,
        session_id=f"eval-{case['id']}",
    )
    try:
        final = await graph.ainvoke(state)
        predictions = {
            g.requirement: g.match_score for g in final.get("gap_analyses", [])
        }
    except Exception as e:
        result.error = str(e)
        return result

    for gold in case.get("requirements", []):
        req_eval = RequirementEval(
            requirement=gold["requirement"],
            gold_bucket=gold["gold_bucket"],
        )
        score = None
        for pred_req, pred_score in predictions.items():
            if _match_requirement(pred_req, gold["requirement"]):
                score = pred_score
                break
        if score is not None:
            req_eval.predicted_score = score
            req_eval.predicted_bucket = score_bucket(score)
            req_eval.bucket_match = req_eval.predicted_bucket == gold["gold_bucket"]
            lo = gold.get("gold_score_min", 1)
            hi = gold.get("gold_score_max", 10)
            req_eval.in_score_range = lo <= score <= hi
        result.requirements.append(req_eval)
    return result


def _layer_metrics(case_results: List[CaseEval]) -> dict:
    """Compute aggregate metrics for a list of case results."""
    all_reqs: List[RequirementEval] = []
    for c in case_results:
        all_reqs.extend(c.requirements)
    scores = [r.predicted_score for r in all_reqs if r.predicted_score is not None]
    gold_buckets = [r.gold_bucket for r in all_reqs if r.predicted_score is not None]

    return {
        "cases_run": len(case_results),
        "requirements_labeled": len(all_reqs),
        "requirements_matched": len(scores),
        "bucket_accuracy": bucket_accuracy(scores, gold_buckets)
        if scores and len(scores) == len(gold_buckets)
        else 0.0,
        "score_in_range_rate": (
            sum(1 for r in all_reqs if r.in_score_range) / len(all_reqs) if all_reqs else 0.0
        ),
    }


def aggregate_report(
    case_results: List[CaseEval], layer_reports: Optional[dict] = None
) -> dict:
    """Build summary report dict from case results, optionally split by layer."""
    overall = _layer_metrics(case_results)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **overall,
        "cases": [
            {
                "case_id": c.case_id,
                "layer": c.layer,
                "job_title": c.job_title,
                "error": c.error,
                "bucket_accuracy": c.bucket_accuracy,
                "requirements": [
                    {
                        "requirement": r.requirement,
                        "gold_bucket": r.gold_bucket,
                        "predicted_score": r.predicted_score,
                        "predicted_bucket": r.predicted_bucket,
                        "bucket_match": r.bucket_match,
                        "in_score_range": r.in_score_range,
                    }
                    for r in c.requirements
                ],
            }
            for c in case_results
        ],
    }
    if layer_reports:
        report["by_layer"] = layer_reports
    return report


async def run_eval(
    case_filter: Optional[str] = None, layer: str = "all"
) -> dict:
    """Run golden eval for selected layer(s) and optional case id filter."""
    datasets = load_datasets(layer)
    all_cases: List[Tuple[str, dict]] = []
    for layer_name, data in datasets:
        errors = validate_dataset(data, layer_name)
        if errors:
            raise ValueError("Dataset invalid: " + "; ".join(errors))
        for case in data["cases"]:
            all_cases.append((layer_name, case))

    if case_filter:
        all_cases = [(ln, c) for ln, c in all_cases if c["id"] == case_filter]
        if not all_cases:
            raise ValueError(f"Unknown case id: {case_filter}")

    builder = WorkflowBuilder()
    results: List[CaseEval] = []
    layer_buckets: dict = {}

    for layer_name, case in all_cases:
        logger.info("Evaluating [%s] case: %s", layer_name, case["id"])
        ev = await evaluate_case(case, builder, layer_name)
        results.append(ev)
        layer_buckets.setdefault(layer_name, []).append(ev)

    layer_reports = {
        name: _layer_metrics(cases) for name, cases in layer_buckets.items()
    }
    return aggregate_report(results, layer_reports if len(layer_reports) > 1 else None)


def _print_dataset_summary(datasets: List[Tuple[str, dict]]):
    """Print case/requirement counts per loaded dataset."""
    for name, data in datasets:
        n_cases = len(data.get("cases", []))
        n_reqs = sum(len(c.get("requirements", [])) for c in data.get("cases", []))
        print(f"  {name}: {n_cases} cases, {n_reqs} labeled requirements")


def main():
    parser = argparse.ArgumentParser(description="Run CareersLow golden-set eval")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset only")
    parser.add_argument(
        "--layer",
        choices=["control", "realism", "all"],
        default="all",
        help="Which dataset layer to run (default: all)",
    )
    parser.add_argument("--case", dest="case_id", help="Run single case id")
    parser.add_argument("--output", help="Write JSON report to path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        datasets = load_datasets(args.layer)
    except FileNotFoundError as e:
        print(str(e))
        raise SystemExit(1)

    print(f"Golden eval layer(s): {args.layer}")
    _print_dataset_summary(datasets)

    all_errors: List[str] = []
    for name, data in datasets:
        all_errors.extend(validate_dataset(data, name))

    if all_errors:
        print("VALIDATION FAILED:")
        for e in all_errors:
            print(f"  - {e}")
        raise SystemExit(1)
    print("Schema validation: OK")

    if args.dry_run:
        print("Dry run complete (no LLM calls).")
        return

    if not os.getenv("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY required for live eval. Use --dry-run to validate only.")
        raise SystemExit(1)

    report = asyncio.run(run_eval(args.case_id, args.layer))
    summary = {k: v for k, v in report.items() if k not in ("cases", "by_layer")}
    print(json.dumps(summary, indent=2))

    if report.get("by_layer"):
        print("\nBy layer:")
        for layer_name, metrics in report["by_layer"].items():
            print(
                f"  {layer_name}: bucket_accuracy={metrics['bucket_accuracy']:.1%}, "
                f"score_in_range={metrics['score_in_range_rate']:.1%}"
            )
    else:
        print(f"\nBucket accuracy: {report['bucket_accuracy']:.1%}")
        print(f"Score in range:  {report['score_in_range_rate']:.1%}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {out}")


if __name__ == "__main__":
    main()
