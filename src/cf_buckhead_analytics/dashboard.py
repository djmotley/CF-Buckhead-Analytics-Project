"""
Streamlit dashboard for the CrossFit Buckhead churn project.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import altair as alt
import pandas as pd
import streamlit as st

from . import config

EXAMPLE_DATA_DIR = Path("data/example_data")


def _latest_file(pattern: str, directory: Path) -> Path | None:
    if not directory.exists():
        return None
    files = sorted(directory.glob(pattern))
    return files[-1] if files else None


def _load_feature_store() -> tuple[pd.DataFrame | None, pd.Timestamp | None]:
    path = _latest_file("feature_store_*.parquet", EXAMPLE_DATA_DIR)
    if not path:
        path = _latest_file("feature_store_*.csv", EXAMPLE_DATA_DIR)
        if not path:
            return None, None
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    as_of = pd.Timestamp(path.stem.split("_")[-1])
    return df, as_of


def _load_training_dataset() -> tuple[pd.DataFrame | None, pd.Timestamp | None]:
    path = _latest_file("training_dataset_*.parquet", EXAMPLE_DATA_DIR)
    if not path:
        return None, None
    df = pd.read_parquet(path)
    as_of = pd.Timestamp(path.stem.split("_")[-1])
    return df, as_of


def _load_model_metadata() -> tuple[str, Path] | None:
    metadata_files = sorted(EXAMPLE_DATA_DIR.glob("model_metadata_*.json"))
    if not metadata_files:
        return None
    latest = metadata_files[-1]
    return latest.read_text(encoding="utf-8"), latest


def _parse_metadata(meta_text_path: tuple[str, Path] | None) -> dict[str, Any] | None:
    if meta_text_path is None:
        return None
    text, path = meta_text_path
    return cast(dict[str, Any], json.loads(text))


def _load_feature_importance(as_of: pd.Timestamp | None) -> pd.DataFrame | None:
    if as_of is not None:
        specific = EXAMPLE_DATA_DIR / f"feature_importance_{as_of.date()}.csv"
        if specific.exists():
            return pd.read_csv(specific)
    fallback = _latest_file("feature_importance_*.csv", EXAMPLE_DATA_DIR)
    if fallback:
        return pd.read_csv(fallback)
    return None


def _load_outreach(as_of: pd.Timestamp | None) -> pd.DataFrame | None:
    if as_of is not None:
        specific = EXAMPLE_DATA_DIR / f"outreach_{as_of.date()}.csv"
        if specific.exists():
            return pd.read_csv(specific)
    fallback = _latest_file("outreach_*.csv", EXAMPLE_DATA_DIR)
    if fallback:
        return pd.read_csv(fallback)
    return None


def _load_cohort_retention(as_of: pd.Timestamp | None) -> pd.DataFrame | None:
    if as_of is not None:
        specific = EXAMPLE_DATA_DIR / f"cohort_retention_{as_of.date()}.csv"
        if specific.exists():
            return pd.read_csv(specific)
    fallback = _latest_file("cohort_retention_*.csv", EXAMPLE_DATA_DIR)
    if fallback:
        return pd.read_csv(fallback)
    return None


def run_dashboard_app() -> None:
    st.set_page_config(page_title="CrossFit Buckhead Retention", layout="wide")
    st.title("CrossFit Buckhead Retention Dashboard")
    st.caption("Monitor churn risk, understand drivers, and prioritize weekly outreach.")

    features_df, feature_as_of = _load_feature_store()
    training_df, training_as_of = _load_training_dataset()
    metadata_text_path = _load_model_metadata()
    metadata = _parse_metadata(metadata_text_path)
    importance_df = _load_feature_importance(training_as_of or feature_as_of)
    outreach_df = _load_outreach(training_as_of or feature_as_of)
    cohort_df = _load_cohort_retention(feature_as_of)

    if features_df is None:
        st.warning(
            "No feature store found. Run `python -m cf_buckhead_analytics.feature_engineering` first."
        )
        st.stop()

    snapshot_ts = training_as_of or feature_as_of
    snapshot_date = snapshot_ts.date().isoformat() if snapshot_ts is not None else "N/A"
    positive_rate = (
        float(training_df["churned"].mean())
        if training_df is not None and not training_df.empty
        else None
    )
    pr_auc = float(metadata["pr_auc"]) if metadata and "pr_auc" in metadata else None
    precision_at_share = (
        float(metadata["precision_at_top_share"])
        if metadata and "precision_at_top_share" in metadata
        else None
    )
    recall_at_share = (
        float(metadata["recall_at_top_share"])
        if metadata and "recall_at_top_share" in metadata
        else None
    )

    st.markdown("### Snapshot KPIs")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Snapshot Date", snapshot_date)
    col2.metric("Positive Rate", f"{positive_rate:.3f}" if positive_rate is not None else "n/a")
    col3.metric(
        f"PR-AUC",
        f"{pr_auc:.3f}" if pr_auc is not None else "n/a",
    )
    col4.metric(
        f"Precision@Top {int(config.CHURN_TOP_SHARE * 100)}%",
        f"{precision_at_share:.3f}" if precision_at_share is not None else "n/a",
    )

    st.markdown("### Tenure Buckets")
    tenure_chart: alt.Chart | None
    if features_df is not None and "tenure_bucket" in features_df.columns:
        tenure_counts = (
            features_df["tenure_bucket"]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .reset_index(name="members")
            .rename(columns={"index": "tenure_bucket"})
        )
        tenure_chart = (
            alt.Chart(tenure_counts)
            .mark_bar()
            .encode(
                x=alt.X("members:Q", title="Members"),
                y=alt.Y("tenure_bucket:N", title="Tenure Bucket"),
                tooltip=["tenure_bucket", "members"],
            )
        )
    else:
        tenure_chart = None

    if tenure_chart is not None:
        st.altair_chart(tenure_chart, use_container_width=True)
    else:
        st.info("Tenure buckets are unavailable for the current feature snapshot.")

    st.markdown("### Plan Mix")
    plan_chart: alt.Chart | None
    if features_df is not None and "plan_norm" in features_df.columns:
        plan_counts = (
            features_df["plan_norm"]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .reset_index(name="members")
            .rename(columns={"index": "plan_norm"})
        )
        plan_chart = (
            alt.Chart(plan_counts)
            .mark_bar()
            .encode(
                x=alt.X("members:Q", title="Members"),
                y=alt.Y("plan_norm:N", title="Plan"),
                tooltip=["plan_norm", "members"],
            )
        )
    else:
        plan_chart = None

    if plan_chart is not None:
        st.altair_chart(plan_chart, use_container_width=True)
    else:
        st.info("Membership plan data not available; display skipped.")

    st.markdown("### Cohort Retention")
    if cohort_df is not None and not cohort_df.empty:
        cohort_df["attendance_month"] = cohort_df["attendance_month"].astype(str)
        heatmap = (
            alt.Chart(cohort_df)
            .mark_rect()
            .encode(
                x=alt.X("months_since_join:O", title="Months Since Join"),
                y=alt.Y("cohort_month:N", title="Cohort"),
                color=alt.Color(
                    "retention_rate:Q", title="Retention", scale=alt.Scale(scheme="blues")
                ),
                tooltip=["cohort_month", "attendance_month", "retention_rate"],
            )
        )
        st.altair_chart(heatmap, use_container_width=True)
    else:
        st.info("Cohort retention matrix unavailable. Will populate once enough history accrues.")

    st.markdown("### Outreach Shortlist")
    if outreach_df is not None and not outreach_df.empty:
        display_cols = [
            "member_id",
            "score",
            "risk_tier",
            "days_since_last_checkin",
            "attend_recent_28",
            "tenure_bucket",
            "plan_norm",
        ]
        available_cols = [col for col in display_cols if col in outreach_df.columns]
        st.dataframe(outreach_df[available_cols], use_container_width=True)
        st.download_button(
            label="Download Outreach CSV",
            data=outreach_df.to_csv(index=False).encode("utf-8"),
            file_name=f"outreach_{snapshot_date}.csv",
        )
    else:
        st.info("Run the churn model to generate an outreach shortlist.")

    st.markdown("### Why These Members?")
    if importance_df is not None and not importance_df.empty:
        top_importance = importance_df.sort_values("importance", ascending=False).head(5)[
            ["feature", "importance"]
        ]
        st.table(top_importance.rename(columns={"feature": "Feature", "importance": "Importance"}))
    else:
        st.info("Train the model to surface feature importance explanations.")


if __name__ == "__main__":
    run_dashboard_app()
