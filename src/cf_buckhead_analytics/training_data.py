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
    if "attend_recent_28" not in base.columns:
        base["attend_recent_28"] = 0

    min_attend = getattr(config, "MIN_RECENT_CHECKINS", 1)
    candidates = base[base["attend_recent_28"] >= min_attend].copy()
    candidates["churned"] = 0
    candidates["churn_event_date"] = pd.NaT
    candidates["churn_event_type"] = "none"

    members = raw.members
    label_mode = "inactivity-based"

    if members is not None and "status" in members.columns:
        members_local = members.copy()
        members_local["status_lower"] = members_local["status"].fillna("").astype(str).str.lower()
        plan_cancel_raw = members_local.get("plan_cancel_date")
        if plan_cancel_raw is not None:
            members_local["plan_cancel_date"] = pd.to_datetime(plan_cancel_raw, errors="coerce")
        else:
            members_local["plan_cancel_date"] = pd.Series(
                pd.NaT, index=members_local.index, dtype="datetime64[ns]"
            )

        plan_end_raw = members_local.get("plan_end_date")
        if plan_end_raw is not None:
            members_local["plan_end_date"] = pd.to_datetime(plan_end_raw, errors="coerce")
        else:
            members_local["plan_end_date"] = pd.Series(
                pd.NaT, index=members_local.index, dtype="datetime64[ns]"
            )
        members_local["label_end_date"] = members_local["plan_cancel_date"].fillna(
            members_local["plan_end_date"]
        )

        merge_cols = members_local[["member_id", "status_lower", "label_end_date"]]
        candidates = candidates.merge(merge_cols, on="member_id", how="left")

        already_cancelled = (
            candidates["status_lower"].eq("cancelled")
            & candidates["label_end_date"].notna()
            & (candidates["label_end_date"] <= as_of)
        )
        if already_cancelled.any():
            candidates = candidates[~already_cancelled].copy()

        lookahead = getattr(config, "CHURN_LOOKAHEAD_DAYS", 30)
        future_mask = (
            candidates["label_end_date"].notna()
            & (candidates["label_end_date"] > as_of)
            & (candidates["label_end_date"] <= as_of + pd.Timedelta(days=lookahead))
        )

        if future_mask.any():
            label_mode = "future-cancel-date"
            candidates.loc[future_mask, "churned"] = 1
            candidates.loc[future_mask, "churn_event_date"] = candidates.loc[
                future_mask, "label_end_date"
            ]
            candidates.loc[future_mask, "churn_event_type"] = "cancellation"

    if label_mode == "inactivity-based":
        print(
            "Labeling fallback triggered: using inactivity >= "
            f"{config.INACTIVITY_DAYS} days as churn signal."
        )
        inactivity_mask = candidates["days_since_last_checkin"] >= config.INACTIVITY_DAYS
        candidates.loc[inactivity_mask, "churned"] = 1
        candidates.loc[inactivity_mask, "churn_event_type"] = "inactivity"

    candidates = candidates.drop(
        columns=[col for col in ["status_lower", "label_end_date"] if col in candidates.columns]
    )
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
