# src/cf_buckhead_analytics/config.py
"""
Single place to tweak project behavior. Everything here is deliberately simple:
- plain constants (UPPERCASE names) for easy reading
- numbers chosen to match our agreed plan
"""

from datetime import time

# Python/runtime
PY_VERSION = "3.13.9"  # recorded for clarity; not enforced here

# ---- Churn definition ----
INACTIVITY_DAYS = 60  # hybrid: canceled OR inactive for 60 days
EARLIEST_EVENT_WINS = True  # don't double-count reason if both occur

# ---- Feature windows ----
FEATURE_LOOKBACK_DAYS = 45  # size of history window when we compute features
MOMENTUM_RECENT_DAYS = 21  # last 3 weeks
MOMENTUM_PRIOR_DAYS = 21  # prior 3 weeks

# ---- Rule-based score weights (must sum to 1.0) ----
WEIGHTS = {
    "recency": 0.35,  # days since last check-in (scaled 0..1)
    "momentum": 0.30,  # drop recent vs prior (0..1)
    "no_show": 0.15,  # no-show/late-cancel rate (0..1)
    "purchase_drop": 0.10,  # drop in purchase cadence (0..1)
    "status": 0.10,  # pending_cancel flag (0 or 1)
}

# ---- Band thresholds for 0..100 score ----
RED_MIN = 70  # RED >= 70
YELLOW_MIN = 55  # 55..69 = YELLOW; <55 = GREEN

# ---- Weekly run time (for docs; scheduling lives elsewhere) ----
WEEKLY_RUN_TIME = time(hour=17, minute=0)  # Sundays 5:00 PM ET

# ---- Paths ----
RAW_DIR = "data/raw"  # real, gitignored
RAW_SAMPLE_DIR = "data/raw_sample_dylan"  # tiny fake CSVs for testing
PROCESSED_DIR = "data/processed"
REPORTS_OUTREACH_DIR = "reports/outreach"
