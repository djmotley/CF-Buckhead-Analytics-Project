# src/cf_buckhead_analytics/risk_scoring.py
"""
Compute churn risk scores from engineered features.
Outputs:
  • scores_<date>.csv        (all members)
  • top10_<date>.csv         (highest-risk outreach list)
"""

from pathlib import Path
from typing import Literal

import pandas as pd

from . import config
from .feature_engineering import build_features


def scale_recency(days: pd.Series, cap: int = config.INACTIVITY_DAYS) -> pd.Series:
    """Convert days-since-checkin into 0-1 risk (0 = today, 1 = ≥cap days)."""
    return (days.clip(lower=0, upper=cap) / cap).astype(float)


def compute_scores(features: pd.DataFrame) -> pd.DataFrame:
    """Apply weighted formula and banding."""
    w = config.WEIGHTS

    recency_risk = scale_recency(features["days_since_last_checkin"])
    momentum_risk = features["momentum_drop"].clip(0, 1)
    purchase_risk = features["purchase_drop"].clip(0, 1)
    status_risk = features["pending_cancel_flag"].clip(0, 1)

    # temporary: no_show not yet implemented
    no_show_risk = 0.0

    score_0_1 = (
        w["recency"] * recency_risk
        + w["momentum"] * momentum_risk
        + w["no_show"] * no_show_risk
        + w["purchase_drop"] * purchase_risk
        + w["status"] * status_risk
    )
    score_0_100 = (score_0_1 * 100).round(1)

    df = features.copy()
    df["score"] = score_0_100

    # Banding thresholds
    def band(x: float) -> Literal["RED", "YELLOW", "GREEN"]:
        if x >= config.RED_MIN:
            return "RED"
        elif x >= config.YELLOW_MIN:
            return "YELLOW"
        else:
            return "GREEN"

    df["band"] = df["score"].apply(band)
    return df


def run(as_of: str | None = None) -> None:
    as_of_date = pd.Timestamp(as_of) if as_of else pd.Timestamp.today().normalize()
    out_dir = Path(config.PROCESSED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = build_features(as_of=as_of_date)
    scored = compute_scores(feats)

    scores_path = out_dir / f"scores_{as_of_date.date()}.csv"
    scored.to_csv(scores_path, index=False)

    # --- Outreach file (top 10 highest risk) ---
    red = scored.query("band == 'RED'").sort_values("score", ascending=False)
    if len(red) < 10:
        yellow_fill = (
            scored.query("band == 'YELLOW'")
            .sort_values("score", ascending=False)
            .head(10 - len(red))
        )
        outreach = pd.concat([red, yellow_fill])
    else:
        outreach = red.head(10)

    reports_dir = Path(config.REPORTS_OUTREACH_DIR)
    reports_dir.mkdir(parents=True, exist_ok=True)
    outreach_path = reports_dir / f"top10_{as_of_date.date()}.csv"
    outreach.to_csv(outreach_path, index=False)

    print(f"Wrote: {scores_path}")
    print(f"Wrote: {outreach_path}")
    print(scored.head())


if __name__ == "__main__":
    run()
