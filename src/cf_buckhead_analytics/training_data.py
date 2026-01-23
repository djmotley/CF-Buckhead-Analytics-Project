# src/cf_buckhead_analytics/training_data.py
"""
Create the supervised training dataset by merging engineered features with
churn labels. Labels are derived from membership status/date when available,
and fall back to inactivity-based churn only when necessary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config
from .data_prep import RawData, ensure_directories, load_raw_data
from .feature_engineering import FeatureArtifacts, generate_feature_store


@dataclass(frozen=True)
class TrainingArtifacts:
    dataset: pd.DataFrame
    as_of: pd.Timestamp
    positive_rate: float
    label_mode: str


def make_training_frame(as_of: str | None = None) -> TrainingArtifacts:
    ensure_directories()

    feature_artifacts: FeatureArtifacts = generate_feature_store(as_of=as_of)
    features = feature_artifacts.features.copy()
    raw = load_raw_data()

    labeled, label_mode = _label_candidates(features, raw, feature_artifacts.as_of)
    positive_rate = float(labeled["churned"].mean()) if not labeled.empty else 0.0

    return TrainingArtifacts(
        dataset=labeled,
        as_of=feature_artifacts.as_of,
        positive_rate=positive_rate,
        label_mode=label_mode,
    )


def save_training_frame(artifacts: TrainingArtifacts) -> Path:
    output_path = Path(config.PROCESSED_DIR) / f"training_dataset_{artifacts.as_of.date()}.parquet"
    artifacts.dataset.to_parquet(output_path, index=False)
    return output_path


def _label_candidates(
    features: pd.DataFrame, raw: RawData, as_of: pd.Timestamp
) -> tuple[pd.DataFrame, str]:
    base = features.copy()
    if "plan_norm" not in base.columns:
        base["plan_norm"] = "Other"
    else:
        base["plan_norm"] = base["plan_norm"].fillna("Other").astype(str).replace({"": "Other"})

    if "attend_recent_28" not in base.columns:
        base["attend_recent_28"] = 0

    min_attend = getattr(config, "MIN_RECENT_CHECKINS", 1)
    candidates = base.copy()
    candidates["churned"] = 0
    candidates["churn_event_date"] = pd.NaT
    candidates["churn_event_type"] = "none"

    members = raw.memberships
    label_mode = "membership-status"

    membership_cols = ["status", "plan_cancel_date", "plan_end_date"]

    missing_cols = [col for col in membership_cols if col not in candidates.columns]
    if members is not None and missing_cols:
        members_local = members.copy()
        merge_cols = [
            col
            for col in ["member_id", "status", "plan_cancel_date", "plan_end_date"]
            if col in members_local.columns
        ]
        membership_join = members_local[merge_cols]
        candidates = candidates.merge(membership_join, on="member_id", how="left")

    status_series = (
        candidates["status"]
        if "status" in candidates.columns
        else pd.Series("", index=candidates.index)
    ).fillna("")
    status_lower = status_series.astype(str).str.lower()

    if "plan_cancel_date" in candidates.columns:
        plan_cancel = pd.to_datetime(candidates["plan_cancel_date"], errors="coerce")
    else:
        plan_cancel = pd.Series(pd.NaT, index=candidates.index, dtype="datetime64[ns]")

    if "plan_end_date" in candidates.columns:
        plan_end = pd.to_datetime(candidates["plan_end_date"], errors="coerce")
    else:
        plan_end = pd.Series(pd.NaT, index=candidates.index, dtype="datetime64[ns]")

    cancel_status_flag = status_lower.eq("cancelled") & (
        (plan_cancel.notna() & (plan_cancel <= as_of))
        | (plan_cancel.isna() & plan_end.notna() & (plan_end < as_of))
    )
    cancel_date_flag = plan_cancel.notna() & (plan_cancel <= as_of)
    plan_end_flag = plan_end.notna() & (plan_end < as_of)
    churn_mask = cancel_status_flag | cancel_date_flag | plan_end_flag

    if members is not None:
        candidates.loc[churn_mask, "churned"] = 1
        candidates.loc[churn_mask, "churn_event_type"] = "membership"
        candidates.loc[churn_mask, "churn_event_date"] = plan_cancel.fillna(plan_end)
    else:
        label_mode = "no-membership-data"
        candidates["status"] = ""
        candidates["plan_cancel_date"] = pd.NaT
        candidates["plan_end_date"] = pd.NaT

    if "plan_norm" in candidates.columns:
        coach_mask = candidates["plan_norm"].astype(str).str.lower() == "coach/staff"
        if coach_mask.any():
            candidates = candidates.loc[~coach_mask].copy()

    candidates = candidates.drop_duplicates("member_id", keep="last").reset_index(drop=True)
    return candidates, label_mode


def _print_summary(artifacts: TrainingArtifacts) -> None:
    df = artifacts.dataset
    snapshot_count = int(df["snapshot_as_of"].nunique()) if not df.empty else 0
    counts = (
        df.groupby("snapshot_as_of")["member_id"].count()
        if not df.empty
        else pd.Series([], dtype=int)
    )
    if not counts.empty:
        min_count = int(counts.min())
        median_count = int(counts.median())
        max_count = int(counts.max())
    else:
        min_count = median_count = max_count = 0

    overall_positive_rate = float(df["churned"].mean()) if not df.empty else 0.0

    print(f"Labeling used: {artifacts.label_mode}; positive rate: {artifacts.positive_rate:.3f}")
    print(f"Snapshots: {snapshot_count}")
    print(f"Candidates per snapshot (min/median/max): {min_count} / {median_count} / {max_count}")
    print(f"Overall positive rate (positives / candidates): {overall_positive_rate:.3f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create churn training dataset.")
    parser.add_argument(
        "--as_of",
        type=str,
        default=None,
        help="Optional snapshot date (YYYY-MM-DD). Defaults to latest available <= today.",
    )
    args = parser.parse_args(argv)

    artifacts = make_training_frame(as_of=args.as_of)
    save_path = save_training_frame(artifacts)
    print(f"Saved training dataset to {save_path}")
    _print_summary(artifacts)


if __name__ == "__main__":
    main()
