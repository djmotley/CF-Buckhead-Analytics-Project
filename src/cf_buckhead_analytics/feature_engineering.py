# src/cf_buckhead_analytics/feature_engineering.py
"""
Feature engineering utilities for the CrossFit Buckhead churn project.

This module transforms the raw attendance (and optional member) data into
per-member feature vectors, and optionally writes cohort retention tables.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data_prep import RawData, ensure_directories, load_raw_data


@dataclass(frozen=True)
class FeatureArtifacts:
    features: pd.DataFrame
    as_of: pd.Timestamp


def generate_feature_store(as_of: str | None = None) -> FeatureArtifacts:
    """Public entry point used by CLI scripts and downstream modules."""

    ensure_directories()
    raw = load_raw_data()
    as_of_ts = _resolve_as_of(raw.attendance, as_of)
    feature_df = _build_feature_matrix(raw, as_of_ts)
    _save_feature_store(feature_df, as_of_ts)
    _maybe_save_cohort_retention(raw.attendance, as_of_ts)
    print(f"Feature snapshot saved: {as_of_ts.date()} … rows: {len(feature_df)}")
    return FeatureArtifacts(features=feature_df, as_of=as_of_ts)


def _resolve_as_of(attendance: pd.DataFrame, as_of: str | None) -> pd.Timestamp:
    if attendance.empty:
        raise ValueError("Attendance data is empty; cannot determine snapshot date.")

    today = pd.Timestamp.today().normalize()
    latest_observed_raw = attendance["class_ts"].max()
    if pd.isna(latest_observed_raw):
        raise ValueError("Attendance data contains no valid timestamps.")
    latest_observed = pd.Timestamp(latest_observed_raw).normalize()

    resolved = latest_observed if latest_observed <= today else today

    override = config.AS_OF_OVERRIDE
    if override is not None:
        override_ts = pd.Timestamp(override).normalize()
        if override_ts < resolved:
            resolved = override_ts

    if as_of is not None:
        explicit_ts = pd.Timestamp(as_of).normalize()
        if explicit_ts < resolved:
            resolved = explicit_ts

    return resolved


def _build_feature_matrix(raw: RawData, as_of: pd.Timestamp) -> pd.DataFrame:
    attendance = raw.attendance.copy()
    attendance = attendance[attendance["class_ts"] <= as_of]

    if attendance.empty:
        return pd.DataFrame(columns=_base_feature_columns())

    grouped = attendance.groupby("member_id")
    first_seen = grouped["class_ts"].min().rename("first_seen")
    last_seen = grouped["class_ts"].max().rename("last_checkin")
    lifetime_attend = grouped.size().rename("lifetime_attend")

    recent_window = as_of - timedelta(days=28)
    prior_start = as_of - timedelta(days=config.FEATURE_LOOKBACK_DAYS)
    prior_end = as_of - timedelta(days=29)

    attend_recent_28 = (
        attendance[attendance["class_ts"].between(recent_window, as_of, inclusive="both")]
        .groupby("member_id")["class_ts"]
        .count()
        .rename("attend_recent_28")
    )

    attend_prior_29_84 = (
        attendance[attendance["class_ts"].between(prior_start, prior_end, inclusive="both")]
        .groupby("member_id")["class_ts"]
        .count()
        .rename("attend_prior_29_84")
    )

    class_breakdown = (
        attendance.pivot_table(
            index="member_id",
            columns="class_type",
            values="class_ts",
            aggfunc="count",
            fill_value=0,
        )
        .rename(columns=_class_type_alias)
        .add_prefix("class_count_")
    )

    feature_df = (
        pd.concat(
            [
                first_seen,
                last_seen,
                lifetime_attend,
                attend_recent_28,
                attend_prior_29_84,
                class_breakdown,
            ],
            axis=1,
        )
        .fillna(0)
        .reset_index()
        .rename(columns={"index": "member_id"})
    )

    feature_df["first_seen"] = pd.to_datetime(feature_df["first_seen"])
    feature_df["last_checkin"] = pd.to_datetime(feature_df["last_checkin"])

    feature_df["tenure_days"] = (as_of - feature_df["first_seen"]).dt.days
    feature_df["days_since_last_checkin"] = (as_of - feature_df["last_checkin"]).dt.days

    if "class_count_Endurance" in feature_df.columns:
        feature_df["class_type_share_endurance"] = (
            feature_df["class_count_Endurance"] / feature_df["lifetime_attend"].replace(0, np.nan)
        ).fillna(0.0)
    else:
        feature_df["class_type_share_endurance"] = 0.0

    bins = [0, 60, 150, 360, 10_000]
    labels = ["New", "Early", "Core", "Loyal"]
    feature_df["tenure_bucket"] = pd.cut(
        feature_df["tenure_days"], bins=bins, labels=labels, include_lowest=True
    ).astype(str)

    feature_df["snapshot_as_of"] = as_of

    if raw.members is not None:
        members = raw.members.copy()
        members = members.drop_duplicates("member_id")
        members["plan_norm"] = members["plan"].apply(_normalize_plan_name)
        members["orig_status"] = members["status"].str.lower()
        feature_df = feature_df.merge(members, on="member_id", how="left")
    else:
        feature_df["plan_norm"] = "Unknown"
        feature_df["status"] = pd.Series([None] * len(feature_df), dtype="object")
        feature_df["plan_cancel_date"] = pd.NaT
        feature_df["plan_end_date"] = pd.NaT
        feature_df["member_since"] = pd.NaT

    feature_df["attend_recent_28"] = feature_df["attend_recent_28"].astype(int)
    feature_df["attend_prior_29_84"] = feature_df["attend_prior_29_84"].astype(int)
    feature_df["lifetime_attend"] = feature_df["lifetime_attend"].astype(int)

    return feature_df[_base_feature_columns() + _optional_member_columns(feature_df)]


def _base_feature_columns() -> list[str]:
    return [
        "member_id",
        "first_seen",
        "last_checkin",
        "tenure_days",
        "days_since_last_checkin",
        "attend_recent_28",
        "attend_prior_29_84",
        "lifetime_attend",
        "class_type_share_endurance",
        "tenure_bucket",
        "snapshot_as_of",
    ]


def _optional_member_columns(df: pd.DataFrame) -> list[str]:
    extras = []
    for col in ["plan_norm", "status", "plan_cancel_date", "plan_end_date", "member_since"]:
        if col in df.columns:
            extras.append(col)
    return extras


def _class_type_alias(name: str) -> str:
    return name.title()


def _normalize_plan_name(plan: str) -> str:
    text = (plan or "").lower()
    if "coach" in text:
        return "Coach"
    if "vip" in text:
        return "VIP"
    if "unlimited" in text or "$189" in text or "$149" in text:
        return "Unlimited"
    if "10" in text or "12" in text or "pack" in text:
        return "Limited"
    return "Other"


def _save_feature_store(features: pd.DataFrame, as_of: pd.Timestamp) -> None:
    date_str = as_of.date().isoformat()
    parquet_path = Path(config.PROCESSED_DIR) / f"feature_store_{date_str}.parquet"
    csv_path = Path(config.PROCESSED_DIR) / f"feature_store_{date_str}.csv"
    features.to_parquet(parquet_path, index=False)
    features.to_csv(csv_path, index=False)


def _maybe_save_cohort_retention(attendance: pd.DataFrame, as_of: pd.Timestamp) -> None:
    if attendance.empty:
        print("Cohort retention skipped: no attendance records.")
        return

    cohorts = attendance.groupby("member_id")["class_ts"].min().dt.to_period("M").rename("cohort")
    attendance = attendance.copy()
    attendance["cohort"] = attendance["member_id"].map(cohorts)
    attendance["attendance_month"] = attendance["class_ts"].dt.to_period("M")

    counts = (
        attendance.groupby(["cohort", "attendance_month"])["member_id"]
        .nunique()
        .reset_index(name="active_members")
    )
    if counts.empty:
        print("Cohort retention skipped: insufficient data.")
        return

    cohort_sizes = counts.groupby("cohort")["active_members"].transform("first")
    counts["cohort_size"] = cohort_sizes
    counts["retention_rate"] = counts["active_members"] / counts["cohort_size"].replace(0, np.nan)

    retention = counts.copy()
    retention["cohort_month"] = retention["cohort"].astype(str)
    retention["months_since_join"] = retention["attendance_month"].astype("int64") - retention[
        "cohort"
    ].astype("int64")
    retention_path = Path(config.PROCESSED_DIR) / f"cohort_retention_{as_of.date().isoformat()}.csv"
    retention[["cohort_month", "attendance_month", "months_since_join", "retention_rate"]].to_csv(
        retention_path, index=False
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build feature store for churn modeling.")
    parser.add_argument(
        "--as_of",
        type=str,
        default=None,
        help="Optional snapshot date (YYYY-MM-DD). Defaults to latest available <= today.",
    )
    args = parser.parse_args(argv)
    generate_feature_store(as_of=args.as_of)


if __name__ == "__main__":
    main()
