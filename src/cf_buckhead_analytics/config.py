from pathlib import Path
from typing import Optional

# === Core Directories ===
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
REPORTS_OUTREACH_DIR = Path("reports/outreach")
MODELS_DIR = Path("models")

# === Feature & Label Windows ===
FEATURE_LOOKBACK_DAYS = 84
RECENT_ACTIVITY_DAYS = 30
INACTIVITY_DAYS = 14
CHURN_TOP_SHARE = 0.20
CHURN_LOOKAHEAD_DAYS = 30  # look ahead this many days for future cancellations
MIN_RECENT_CHECKINS = 1  # require at least this many classes in last 28 days

# === Scheduling ===
AS_OF_OVERRIDE: Optional[str] = "2024-09-28"  # YYYY-MM-DD or None

# === Model Settings ===
RANDOM_STATE = 42

# === Legacy Guards ===
ALLOW_SYNTHETIC_LABELS = False  # never synthesize churn labels

# === Outreach Settings ===
OUTREACH_MAX_MEMBERS = 15
