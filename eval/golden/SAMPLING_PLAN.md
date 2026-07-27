# Realism eval sampling plan

## Goal

Build **Layer 2** golden set from licensed public resume+JD pairs, re-labeled for CareersLow's strong/weak/gap rubric.

Layer 1 (`dataset.json`) stays as regression **control**. Layer 2 (`dataset_realism.json`) tests parser noise, messy prose, and generalization.

---

## Source datasets (use these, not web scrape)

| Dataset | URL | License | Use for |
|---------|-----|---------|---------|
| **job_resume_fit** | [HuggingFace](https://huggingface.co/datasets/batuhanmtl/job_resume_fit) | Kaggle terms | Primary pair pool (2,385 rows, 23 categories) |
| **candidate-matching-synthetic** | [HuggingFace](https://huggingface.co/datasets/michaelozon/candidate-matching-synthetic) | MIT | Backup pool, seniority-aligned pairs |
| **JobResQA** | [GitHub](https://github.com/Avature/jobresqa-benchmark) | Open benchmark | Future debrief/Q&A eval |

**Do not** scrape LinkedIn/Indeed or copy real candidate PII.

---

## Selection criteria (pick 20–30 pairs)

For each candidate row, score on:

1. **Role diversity** — at least 2 cases per persona we care about (backend, frontend, data, DevOps, mobile, QA)
2. **Skill mismatch variety** — mix of high-fit, partial-fit, and clear mismatch pairs (not all 90%+ AI scores)
3. **JD messiness** — prefer rows with long `job_text` (boilerplate, benefits, EEO) over tiny skill lists
4. **Resume realism** — varied bullet styles, some vague wording ("cloud platforms", "agile environment")
5. **Parser alignment** — after running CareersLow JD parser, ≥4 extractable requirements per case

**Reject** if:
- Resume or JD under 80 words (too thin)
- Duplicate category already over-represented (>4 cases)
- Cannot identify at least one clear gap and one clear strength

---

## Labeling workflow

```
Public pair  →  Run JD parser only  →  Human labels parsed requirements  →  dataset_realism.json
```

Per case (~10 min):

1. Copy `resume_text` + `job_text` into a draft case
2. Run: `python -m src.eval.parse_jd_preview --jd "..."` (or full pipeline dry inspect)
3. List requirements the **parser actually outputs**
4. For each requirement assign:
   - `gold_bucket`, `gold_score_min/max`
   - `relevant_resume_snippets`
   - `label_notes` (cite resume evidence)
5. Set metadata:
   - `layer`: `"realism"`
   - `source`: `"batuhanmtl/job_resume_fit#<category>:<row_hint>"`
   - `source_category`: e.g. `"Data Science"`

**Never** copy `ai_match_score` from Kaggle as gold — it measures skill overlap, not per-requirement buckets.

---

## Rollout phases

| Phase | Cases | Requirements | Status |
|-------|-------|--------------|--------|
| 0 | 15 realism seed | 75 | **Done** — `dataset_realism.json` |
| 1 | +10 from job_resume_fit | +50 | Sample script + manual label |
| 2 | +10 synthetic (MIT) | +50 | Hard cases, seniority mismatch |
| 3 | 10 PDF conversions | — | Upload-path smoke tests |

Target: **~175 labeled requirements** across both layers (50 control + 75 realism + future growth).

---

## Sampling script (Phase 1)

```bash
# Requires: pip install datasets
python scripts/sample_realism_pairs.py --dataset batuhanmtl/job_resume_fit --n 15 --out eval/golden/candidates.json
```

Script filters by category diversity and skill mismatch, outputs **candidates for human labeling** (not auto-labeled gold).

---

## Eval reporting

```bash
python -m src.eval.run_golden_eval --layer control    # dataset.json only
python -m src.eval.run_golden_eval --layer realism      # dataset_realism.json only
python -m src.eval.run_golden_eval --layer all          # both, split metrics
```

**Interpretation:**

| Pattern | Likely cause |
|---------|--------------|
| control high, realism low | JD parser or retrieval — not rubric |
| both low | Scoring prompt / bucket definitions |
| realism high, control low | Label bug in control set — review |

---

## PDF stress subset (Phase 3)

Convert 10 realism resumes to PDF via:
- LibreOffice / `reportlab` from plain text
- Or Harvard/MIT public resume PDF templates (educational use)

Store under `eval/golden/pdfs/` and add `pdf_path` field to case schema.
