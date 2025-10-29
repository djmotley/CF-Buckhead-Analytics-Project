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
    memberships = data["memberships"].copy()
    members = data["members"].copy()
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
    memberships = memberships.set_index("member_id")
    pending_flag = (
        memberships["pending_cancel"].fillna(False).astype(int).rename("pending_cancel_flag")
    )

    # ---- Combine all features ----
    features = pd.concat(
        [days_since_last, momentum_drop, purchase_drop, pending_flag], axis=1
    ).fillna(0)
    features.index.name = "member_id"

    # ---- Membership age (days) ----
    members = members.set_index("member_id")
    members["joined_on"] = pd.to_datetime(members["joined_on"], errors="coerce")
    membership_age = (as_of - members["joined_on"]).dt.days.rename("membership_age_days")
    features = features.join(membership_age)

    # ---- Attendance streak (consecutive days up to latest visit) ----
    if not att.empty:
        att_sorted = att.sort_values(["member_id", "date"])
        prev_dates: pd.Series = att_sorted.groupby("member_id")["date"].shift()
        att_sorted["prev_date"] = prev_dates
        gap: pd.Series = att_sorted["date"] - prev_dates
        att_sorted["gap_days"] = gap.dt.days
        att_sorted["streak_break"] = (att_sorted["gap_days"] != 1).fillna(True).astype(int)
        att_sorted["streak_group"] = att_sorted.groupby("member_id")["streak_break"].cumsum()
        att_sorted["streak_length"] = (
            att_sorted.groupby(["member_id", "streak_group"]).cumcount() + 1
        )
        latest_streak = att_sorted.groupby("member_id")["streak_length"].last()
        features = features.join(latest_streak.rename("attendance_streak"))
    else:
        features["attendance_streak"] = 0

    features["membership_age_days"] = features["membership_age_days"].fillna(0).astype("Int64")
    features["attendance_streak"] = features["attendance_streak"].fillna(0).astype("Int64")

    return features.reset_index()


if __name__ == "__main__":
    feats = build_features()
    print(feats.head())
