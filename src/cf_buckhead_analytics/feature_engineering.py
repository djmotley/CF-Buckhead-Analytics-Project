# src/cf_buckhead_analytics/feature_engineering.py
"""
Builds analytic features from the raw sample data (attendance, memberships, members, transactions).
"""

from datetime import timedelta
from pathlib import Path

import pandas as pd

from . import config
from .data_prep import load_sample_data


def build_features(as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Compute member-level features up to the given as-of date."""
    as_of = pd.Timestamp(as_of) if as_of else pd.Timestamp.today().normalize()
    data = load_sample_data(Path(config.RAW_SAMPLE_DIR))  # <- using Dylan's folder

    att = data["attendance"].copy()
    mem = data["memberships"].copy()
    tx = data["transactions"].copy()

    # ---- Attendance-based features ----
    att["date"] = pd.to_datetime(att["date"])
    last_checkin = att.groupby("member_id")["date"].max()
    days_since_last = (as_of - last_checkin).dt.days.rename("days_since_last_checkin")

    # attendance counts in recent vs prior 3-week windows
    recent_start = as_of - timedelta(days=config.MOMENTUM_RECENT_DAYS)
    prior_start = recent_start - timedelta(days=config.MOMENTUM_PRIOR_DAYS)
    recent_mask = (att["date"] >= recent_start) & (att["date"] < as_of)
    prior_mask = (att["date"] >= prior_start) & (att["date"] < recent_start)
    recent_cnt = att.loc[recent_mask].groupby("member_id")["date"].count()
    prior_cnt = att.loc[prior_mask].groupby("member_id")["date"].count()
    momentum_drop = ((prior_cnt - recent_cnt) / prior_cnt.replace(0, pd.NA)).clip(lower=0).fillna(0)
    momentum_drop.name = "momentum_drop"

    # ---- Purchase cadence drop ----
    tx["date"] = pd.to_datetime(tx["date"])
    tx_recent_mask = (tx["date"] >= recent_start) & (tx["date"] < as_of)
    tx_prior_mask = (tx["date"] >= prior_start) & (tx["date"] < recent_start)
    t_recent = tx.loc[tx_recent_mask].groupby("member_id")["date"].count()
    t_prior = tx.loc[tx_prior_mask].groupby("member_id")["date"].count()
    purchase_drop = ((t_prior - t_recent) / t_prior.replace(0, pd.NA)).clip(lower=0).fillna(0)
    purchase_drop.name = "purchase_drop"

    # ---- Pending cancel flag ----
    mem = mem.set_index("member_id")
    pending_flag = mem["pending_cancel"].fillna(False).astype(int).rename("pending_cancel_flag")

    # ---- Combine all features ----
    features = pd.concat(
        [days_since_last, momentum_drop, purchase_drop, pending_flag], axis=1
    ).fillna(0)
    features.index.name = "member_id"
    return features.reset_index()


if __name__ == "__main__":
    feats = build_features()
    print(feats.head())
