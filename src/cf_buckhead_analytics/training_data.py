# src/cf_buckhead_analytics/training_data.py
"""
Build a simple training dataset: engineered features + a basic churn label.
Label rule (simple): churned = 1 if member has a cancellation date, else 0.
"""

from pathlib import Path

import pandas as pd

from . import config
from .data_prep import load_sample_data
from .feature_engineering import build_features


def make_training_frame(as_of: str | None = None) -> pd.DataFrame:
    """
    Returns a DataFrame with:
      - features: days_since_last_checkin, momentum_drop, purchase_drop, pending_cancel_flag
      - label: churned (0/1)
    """
    # features up to as_of
    as_of_ts = pd.Timestamp(as_of) if as_of else None
    feats = build_features(as_of=as_of_ts)

    # load memberships to derive the label
    data = load_sample_data(Path(config.RAW_SAMPLE_DIR))
    mem = data["memberships"][["member_id", "canceled_on"]].copy()
    mem["churned"] = mem["canceled_on"].notna().astype(int)

    # join features with label
    df = feats.merge(mem[["member_id", "churned"]], on="member_id", how="left")
    df["churned"] = df["churned"].fillna(0).astype(int)

    # optional: keep only the columns we need (order for readability)
    cols = [
        "member_id",
        "days_since_last_checkin",
        "momentum_drop",
        "purchase_drop",
        "pending_cancel_flag",
        "membership_age_days",
        "attendance_streak",
        "churned",
    ]
    return df[cols]


if __name__ == "__main__":
    train = make_training_frame()
    print(train.head())
    print("\nClass balance (churned counts):")
    print(train["churned"].value_counts(dropna=False))
