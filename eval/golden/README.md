# Golden evaluation dataset

Two-layer eval strategy — see `SAMPLING_PLAN.md` for expansion plan.

| Layer | File | Purpose |
|-------|------|---------|
| **control** | `dataset.json` | Curated regression anchor (10 cases, 50 reqs) |
| **realism** | `dataset_realism.json` | Messy public-style pairs (15 cases, 75 reqs) |

## What is labeled (not random)

Each **case** is a curated `(resume, JD)` pair. Requirements are **hand-labeled** by design:

| Field | Purpose |
|-------|---------|
| `gold_bucket` | Human judgment: `strong` (8–10), `weak` (5–7), `gap` (1–4) |
| `gold_score_min/max` | Acceptable range for automated `match_score` |
| `relevant_resume_snippets` | Text that Qdrant retrieval should surface for this requirement |
| `duty_requirements` | Optional duty themes (experience/project retrieval only) |
| `label_notes` | Rationale for QA reviewers |
| `layer` | `control` or `realism` |
| `source` | Provenance (e.g. `inspired_by:batuhanmtl/job_resume_fit#Data Science`) |

Cases cover different personas (backend, frontend, ML, DevOps, etc.) with **intentional** strengths and gaps — not random text.

## Files

- `dataset.json` — Layer 1 control set
- `dataset_realism.json` — Layer 2 realism seed
- `schema.json` — JSON schema for one case
- `SAMPLING_PLAN.md` — How to grow realism set from HuggingFace
- `../scripts/sample_realism_pairs.py` — Sample candidates for labeling

## Run local eval (needs GOOGLE_API_KEY)

```bash
python -m src.eval.run_golden_eval --dry-run
python -m src.eval.run_golden_eval --layer control
python -m src.eval.run_golden_eval --layer realism
python -m src.eval.run_golden_eval --layer all --output eval/reports/latest.json
python -m src.eval.run_golden_eval --case realism_data_scientist_01
```

## Targets

See `eval/GOLDEN_EVAL_TEST_PLAN.md`.
