"""
Streamlit dashboard focused on the latest churn snapshot.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

try:
    from . import config
except ImportError:  # pragma: no cover - streamlit executes file directly
    import sys

    PACKAGE_ROOT = Path(__file__).resolve().parents[1]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.append(str(PACKAGE_ROOT))
    import cf_buckhead_analytics.config as config  # pragma: no cover


def _latest_artifact(pattern: str, directory: Path) -> Path | None:
    if not directory.exists():
        return None
    files = sorted(directory.glob(pattern))
    return files[-1] if files else None


def _load_feature_store() -> tuple[pd.DataFrame | None, pd.Timestamp | None]:
    path = _latest_artifact("feature_store_*.parquet", config.PROCESSED_DIR)
    if path is None:
        return None, None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None, None
    snapshot = pd.Timestamp(path.stem.split("_")[-1])
    return df, snapshot


def _load_training_dataset() -> pd.DataFrame | None:
    path = _latest_artifact("training_dataset_*.parquet", config.PROCESSED_DIR)
    if path is None:
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _load_model_metadata() -> tuple[dict[str, Any] | None, pd.Timestamp | None]:
    path = _latest_artifact("model_metadata_*.json", config.MODELS_DIR)
    if path is None:
        return None, None
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    snapshot = pd.Timestamp(path.stem.split("_")[2])
    return metadata, snapshot


def _load_outreach(as_of: pd.Timestamp | None) -> pd.DataFrame | None:
    if as_of is None:
        return None
    path = config.REPORTS_OUTREACH_DIR / f"outreach_{as_of.date()}.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _load_feature_importance(as_of: pd.Timestamp | None) -> pd.DataFrame | None:
    if as_of is None:
        return None
    path = config.PROCESSED_DIR / f"feature_importance_{as_of.date()}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "feature" not in df.columns or "importance" not in df.columns:
        return None
    return df


def _format_metric(value: Any, default: str = "Not available") -> str:
    """Format a numeric metric for display, with a default fallback."""
    try:
        numeric = float(value)
        if pd.notna(numeric):
            return f"{numeric:.3f}"
    except (TypeError, ValueError):
        pass
    return default


def run_dashboard_app() -> None:
    st.set_page_config(page_title="CrossFit Buckhead Churn", layout="wide")
    st.title("CrossFit Buckhead Churn Outlook")
    st.caption("Here is who to contact, and why.")

    features_df, feature_snapshot = _load_feature_store()
    training_df = _load_training_dataset()
    metadata, metadata_snapshot = _load_model_metadata()

    if features_df is None or features_df.empty:
        st.warning("Feature store unavailable. Run feature engineering to refresh the snapshot.")
        return

    snapshot_ts = metadata_snapshot or feature_snapshot
    snapshot_label = snapshot_ts.date().isoformat() if snapshot_ts is not None else "Not available"

    outreach_df = _load_outreach(snapshot_ts)
    feature_importance_df = _load_feature_importance(snapshot_ts)

    attend_recent = pd.to_numeric(
        features_df.get("attend_recent_28", pd.Series(dtype=float)), errors="coerce"
    )
    attend_recent = attend_recent.fillna(0)
    active_members = int((attend_recent > 0).sum())

    outreach_count = int(len(outreach_df)) if outreach_df is not None else 0

    pr_auc = metadata.get("pr_auc") if metadata else None
    precision_at_top = metadata.get("precision_at_top_share") if metadata else None
    recall_at_top = metadata.get("recall_at_top_share") if metadata else None

    metrics = st.columns(5)
    metrics[0].metric("Snapshot Date", snapshot_label)
    metrics[1].metric("Active Members", f"{active_members:,}")
    metrics[2].metric("At-Risk Members", f"{outreach_count:,}")
    metrics[3].metric("PR-AUC", _format_metric(pr_auc))
    metrics[4].metric("Precision@Top 20%", _format_metric(precision_at_top))
    st.metric("Recall@Top 20%", _format_metric(recall_at_top))

    st.markdown("### Feature Importance")
    if feature_importance_df is None or feature_importance_df.empty:
        st.info("Feature importance unavailable.")
    else:
        importance_df = (
            feature_importance_df[["feature", "importance"]]
            .dropna()
            .sort_values("importance", ascending=False)
            .head(15)
        )
        chart = (
            alt.Chart(importance_df)
            .mark_bar()
            .encode(
                x=alt.X("importance:Q", title="Importance"),
                y=alt.Y("feature:N", title="Feature", sort="-x"),
                tooltip=[
                    alt.Tooltip("feature:N", title="Feature"),
                    alt.Tooltip("importance:Q", title="Importance", format=".3f"),
                ],
            )
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("### Outreach List")
    if outreach_df is None or outreach_df.empty:
        st.info("Outreach list unavailable. Run churn regression to generate outreach scores.")
    else:
        outreach_df = outreach_df.copy()
        if "score" in outreach_df.columns:
            outreach_df["score"] = pd.to_numeric(outreach_df["score"], errors="coerce")
            outreach_df = outreach_df.sort_values("score", ascending=False)
        columns = [
            "member_id",
            "score",
            "risk_tier",
            "days_since_last_checkin",
            "attend_recent_28",
            "tenure_bucket",
            "plan_norm",
        ]
        available = [col for col in columns if col in outreach_df.columns]
        st.dataframe(outreach_df[available], use_container_width=True)
        csv_data = outreach_df.to_csv(index=False)
        st.download_button(
            label="Download Outreach CSV",
            data=csv_data,
            file_name=f"outreach_{snapshot_label}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    run_dashboard_app()
