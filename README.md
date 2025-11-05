# CrossFit Buckhead Retention Analytics

**A portfolio-ready project that shows how Derek Motley uses data to spot churn risk early, tailor outreach, and quantify the impact of retention work at a CrossFit gym.**

---

## Why This Matters

Early churn hurts recurring revenue and morale. The Buckhead gym owner wanted a weekly signal that highlights which members are drifting before they cancel, plus a tight feedback loop that shows whether “white‑glove” outreach is paying off. The only guaranteed data set is a PushPress attendance export (`data/raw/Attendance.csv`), and occasionally a roster export (`Members.csv`). Everything in this repo is designed to run—even if the member file is missing—while still delivering a compelling story to admissions committees and hiring managers.

---

## Acquisition Shock: January–February 2025

During admissions season the gym acquired another box. The member base doubled overnight, tenure distributions shifted, and the old churn model—trained on “Original” members—started flagging the wrong people. The pipeline now detects (and gracefully survives) this kind of shock:

- **Lifecycle buckets** adjust automatically: `New (0–2 months)`, `Early (2–5)`, `Core (6–11)`, `Loyal (12+)`.
- **Origin awareness** appears only when it can be inferred (Original vs. New vs. Acquired); otherwise the code continues without it.
- **Time-bounded labeling** keeps us honest: we only call someone churned when a real cancellation or inactivity event occurs within the configured window.

---

## Data Reality (and Fallbacks)

| File | Required | Key Columns | Notes |
|------|----------|-------------|-------|
| `data/raw/Attendance.csv` | ✅ | `member_id`, `class_ts`, `class_type` | Future timestamps are clipped to “today”. Missing `class_type` becomes “CrossFit”. |
| `data/raw/Members.csv` | optional | `member_id`, `status`, `plan`, `memberSince`, `planCancelDate`, `planEndDate` (varies) | When present, the loader normalizes column names, parses dates, and derives missing fields (e.g., `member_since` falls back to first attendance). When absent, the pipeline falls back to behavior-only churn. |

No other files are assumed. Old references to `Cancellations.csv`, `Store_Sales.csv`, or “FINAL” exports have been removed.

---

## Pipeline Overview

### 1. Ingestion (`data_prep.py`)
- Normalizes headers (`Member_ID`, `memberID`, etc.) and ensures `member_id` is always a string.
- Clips future attendance timestamps to “today” while preserving time-of-day.
- Loads optional member roster, standardizes status/plan fields, converts plan end/cancel dates, and fills missing `member_since` from attendance.

### 2. Feature Store (`feature_engineering.py`)
- Snapshot date (`as_of`) defaults to the latest attendance date ≤ today (honors `config.AS_OF_OVERRIDE` or CLI `--as_of`).
- Per-member features include: `first_seen`, `last_checkin`, `tenure_days`, `days_since_last_checkin`, `attend_recent_28`, `attend_prior_29_84`, `lifetime_attend`, class-type mix (Endurance share), and `tenure_bucket`.
- Optional enrichments: plan normalization (`Unlimited`, `Limited`, `Coach`, `VIP`, `Other`) and `member_since` when roster data exists.
- Saves both `.parquet` and `.csv` feature stores plus a cohort-retention matrix (`cohort_retention_<as_of>.csv`) when enough history exists.

Command:
```bash
python -m cf_buckhead_analytics.feature_engineering --as_of 2025-10-01
```
Output example:
```
Feature snapshot saved: 2025-10-01 … rows: 508
```

### 3. Labeling (`training_data.py`)
- Candidates = members with `attend_recent_28 > 0` (someone we can still save).
- Label priority:
  1. **Status-based**: `status.lower() == "cancelled"`.
  2. **Date-based**: plan cancel/end date within `INACTIVITY_DAYS` (default 60) of the snapshot.
  3. **Inactivity fallback**: `days_since_last_checkin >= 60` (prints a clear message when this kicks in).
- Prints labeling mode, candidate counts (min/median/max), and positive rate. Saves `training_dataset_<as_of>.parquet`.

Command:
```bash
python -m cf_buckhead_analytics.training_data --as_of 2025-10-01
```
Output excerpt:
```
Labeling used: status-based; positive rate: 0.147
Snapshots: 1
Candidates per snapshot (min/median/max): 512 / 512 / 512
Overall positive rate (positives / candidates): 0.147
Saved training dataset to data/processed/training_dataset_2025-10-01.parquet
```

### 4. Modeling (`churn_regression.py`)
- Drops plan_norm == “Coach” to keep the focus on member-facing plans.
- Features: `days_since_last_checkin`, `attend_recent_28`, `attend_prior_29_84`, `lifetime_attend`, `tenure_days`, plus categorical `tenure_bucket`, `plan_norm`.
- HistGradientBoostingClassifier (scikit-learn) with stratified hold-out and permutation importance (when both classes present).
- Saves model artifact (`models/churn_model_<as_of>.pkl`), metadata JSON (PR-AUC, precision/recall at top share, sample counts), feature importance CSV, and `reports/outreach/outreach_<as_of>.csv` containing top 15 at-risk members with risk tiers.

Command:
```bash
python -m cf_buckhead_analytics.churn_regression --as_of 2025-10-01
```
Output excerpt:
```
Training samples: 512 (positive rate 0.147)
Top feature drivers:
  - days_since_last_checkin: 0.0821
  - attend_recent_28: 0.0576
  ...
PR-AUC: 0.712; Precision@Top 20%: 0.643; Recall@Top 20%: 0.512; Saved outreach to reports/outreach/outreach_2025-10-01.csv
```

### 5. Dashboard (`dashboard.py`)
- Launch with:
  ```bash
  streamlit run src/cf_buckhead_analytics/dashboard.py
  ```
- Displays KPIs (snapshot, positive rate, PR-AUC, precision at top share), tenure and plan mix bar charts, cohort retention heatmap (if available), outreach shortlist (top 10–15) with download, and “Why these members?” top feature drivers.

---

## Metrics That Matter

- **PR-AUC (Average Precision):** gauges ranking quality when churners are rare. Accuracy would overstate performance.
- **Precision & Recall @ Top 20%:** reflects the owner’s capacity—15 outreach slots per week. High precision means fewer wasted calls; recall shows coverage of true churners.
- **Positive Rate in Training Data:** keeps us honest about class imbalance. If status/cancel data disappears, the fallback prints a warning so stakeholders know we’re on an inactivity proxy.

---

## Weekly Operational Loop

1. **Drop new CSVs** into `data/raw/`.
2. **Rebuild artifacts**:
   ```bash
   python -m cf_buckhead_analytics.feature_engineering
   python -m cf_buckhead_analytics.training_data
   python -m cf_buckhead_analytics.churn_regression
   ```
3. **Review the dashboard** for onboarding trends, plan mix, cohort retention, and the outreach shortlist.
4. **Download outreach CSV** and personalize contact scripts around the top drivers (e.g., “you’ve missed the last three weeks—need help booking classes?”).
5. **Log outcomes** back into PushPress; the next week’s run captures behavior changes automatically.

---

## What’s Next

- **PushPress API integration** to eliminate manual CSV handling and capture membership status changes in real time.
- **Expanded behavioral signals** (no-show penalties, coach check-ins, merchandise purchases) to refine the feature store.
- **Uplift modeling** to prioritize members whose retention probability increases the most after outreach.
- **Automated scheduling** (e.g., GitHub Actions) to refresh scores every Sunday night and email the outreach CSV to the owner.

---

## How to Reproduce

```bash
pip install -e .
pre-commit run --all-files
pytest

python -m cf_buckhead_analytics.feature_engineering
python -m cf_buckhead_analytics.training_data
python -m cf_buckhead_analytics.churn_regression
streamlit run src/cf_buckhead_analytics/dashboard.py
```

The sample data in `data/raw_sample_dylan/` is included for smoke testing, but the project is meant to run on real PushPress exports placed in `data/raw/`.

---

## About Derek Motley

CrossFit Level 1 Trainer, aspiring analytics leader, and the storyteller behind this project. Derek blends coaching insight with statistical rigor to help boutique gyms retain members and grow sustainably.
