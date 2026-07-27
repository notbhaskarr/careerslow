# Golden evaluation test plan

## Purpose

Measure **gap analysis quality** against human-labeled `(resume, JD, requirement)` examples in `eval/golden/dataset.json`.

This is **not random data**. Each of 10 cases has 5 requirements with intentional `strong` / `weak` / `gap` labels and score ranges.

---

## Dataset contents

### Layer 1 — Control (`dataset.json`)

10 curated cases, 50 labeled requirements. Regression anchor.

### Layer 2 — Realism (`dataset_realism.json`)

5 seed cases, 25 labeled requirements. Inspired by [job_resume_fit](https://huggingface.co/datasets/batuhanmtl/job_resume_fit) categories with messy JD boilerplate.

| Case ID | Category | Notes |
|---------|----------|-------|
| `realism_data_scientist_01` | Data Science | BI analyst vs ML engineer JD |
| `realism_java_backend_02` | Java Developer | Spring Boot strong; K8s gap |
| `realism_react_frontend_03` | React Developer | React strong; a11y/GraphQL gap |
| `realism_devops_sre_04` | DevOps | AWS strong; K8s gap |
| `realism_hr_mismatch_05` | HR | Clear role mismatch (HR vs TPM) |
| `realism_python_backend_06` | Python Developer | Django strong; Kafka/K8s gap |
| `realism_node_fullstack_07` | Full Stack | Node/React strong; Next/Postgres gap |
| `realism_android_mobile_08` | Android Developer | Kotlin strong; Compose gap |
| `realism_cloud_architect_09` | Cloud Architect | AWS strong; Azure/K8s gap |
| `realism_qa_automation_10` | QA Engineer | Selenium strong; Playwright gap |
| `realism_cybersecurity_11` | Cybersecurity | SIEM strong; appsec gap |
| `realism_business_analyst_12` | Business Analyst | BA skills strong; insurance gap |
| `realism_dba_postgres_13` | Database Admin | Postgres DBA strong; Citus gap |
| `realism_ml_engineer_14` | ML Engineer | Tabular ML strong; CV/PyTorch gap |
| `realism_network_engineer_15` | Network Engineer | Cisco strong; AWS cloud gap |

Grow to 20–30 cases via `eval/golden/SAMPLING_PLAN.md`.

---

## Test levels

### Level 0 — Schema (no API keys, no cost)

```bash
python -m src.eval.run_golden_eval --dry-run --layer all
pytest tests/test_golden_dataset.py -v
```

**Pass:** both layers validate; control 10/50, realism 15/75.

### Level 1 — Unit metrics (no API keys)

```bash
pytest tests/test_eval_metrics.py -v
```

**Pass:** precision@k, bucket_accuracy formulas correct.

### Level 2 — Live golden eval (needs GOOGLE_API_KEY, Qdrant, Redis)

```bash
docker compose up -d
python -m src.eval.run_golden_eval --layer all --output eval/reports/latest.json
python -m src.eval.run_golden_eval --layer control
python -m src.eval.run_golden_eval --layer realism
```

**Pass criteria (initial targets):**

| Metric | Target | Meaning |
|--------|--------|---------|
| `bucket_accuracy` (control) | ≥ **75%** | Rubric still correct |
| `bucket_accuracy` (realism) | ≥ **65%** | Generalizes to messy inputs |
| `score_in_range_rate` | ≥ **70%** | Numeric score within gold band |
| `requirements_matched` | ≥ **90%** | Pipeline returned a score for labeled requirement |

**Interpretation:** control high + realism low → fix JD parser/retrieval. Both low → fix scoring rubric.

**Review failures manually:** open `eval/reports/latest.json` → cases with `bucket_match: false`.

### Level 3 — LangSmith (optional)

```bash
# Upload dataset cases to LangSmith UI or script, then:
python -m src.eval.evaluate
```

**Pass:** `is_grounded` ≥ 0.9, `jargon_free` ≥ 0.85 on bullet generation.

### Level 4 — E2E product (manual + Redis panel)

See main QA checklist in app: analyze → mock → debrief → **Redis Debug** panel.

---

## How to interpret scores

| Metric | Good | Investigate |
|--------|------|-------------|
| bucket_accuracy 75%+ | Labels align with model | &lt; 60% — rubric or parsing issue |
| score_in_range 70%+ | Numeric calibration OK | Wide misses on gap items — retrieval issue |
| requirements_matched &lt; 90% | JD parser missing skills | Fix `parse_jd` / requirement extraction |

---

## Adding new golden cases

1. Copy a case block in `dataset.json`
2. Write resume + JD so labels are **unambiguous**
3. Assign buckets before running pipeline (human first)
4. Run `--case your_new_id` and tune prompts/rubric
5. Do not add random lorem ipsum — every label needs `label_notes`

---

## Redis validation (in-app)

After analyze/mock, open **Redis Debug** in the UI:
- Confirm `analysis:{pair_id}` exists after analyze
- Confirm `session_meta:{session_id}` phases match interview heard
- Confirm `debrief:{session_id}` after mock ends

---

## Report artifacts

| Path | Contents |
|------|----------|
| `eval/reports/latest.json` | Full per-case eval from `run_golden_eval` |
| LangSmith experiments | LLM judge scores over time |
