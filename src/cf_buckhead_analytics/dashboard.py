# src/cf_buckhead_analytics/dashboard.py
"""
Streamlit dashboard for CrossFit Buckhead churn analysis.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from cf_buckhead_analytics import config

st.set_page_config(page_title="CrossFit Buckhead Member Retention", layout="wide")

st.title(" CrossFit Buckhead Member Retention Dashboard")
st.markdown("Visual overview of member engagement and churn risk scores.")

# --- load score files ---
processed_dir = Path(config.PROCESSED_DIR)
files = sorted(processed_dir.glob("scores_*.csv"))

if not files:
    st.error("No score files found. Run risk_scoring.py first to generate one.")
    st.stop()

# dropdown for file selection
file_names = [f.name for f in files]
selected_file = st.selectbox("Select score file to view:", file_names, index=len(file_names) - 1)

df = pd.read_csv(processed_dir / selected_file)
st.caption(f"Currently viewing: `{selected_file}`")

# preview
st.dataframe(df.head())

# --- refresh button to rerun analysis ---
st.subheader("🔄 Refresh Data")

if st.button("Run Latest Churn Analysis"):
    from cf_buckhead_analytics.risk_scoring import run as run_risk_scoring

    with st.spinner("Running churn pipeline..."):
        run_risk_scoring()
    st.success("✅ Analysis complete! Refresh the dropdown above to see new results.")

# --- summary metrics ---
total_members = len(df)
green_count = (df["band"] == "GREEN").sum()
yellow_count = (df["band"] == "YELLOW").sum()
red_count = (df["band"] == "RED").sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Members", total_members)
col2.metric("Green (Healthy)", green_count)
col3.metric("Yellow (Watch)", yellow_count)
col4.metric("Red (At Risk)", red_count)

# --- Risk score distribution (Seaborn + Matplotlib) ---
st.subheader("Risk Score Distribution")

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(
    data=df,
    x="score",
    bins=20,
    hue="band",
    multiple="stack",
    palette={"GREEN": "#2ecc71", "YELLOW": "#f1c40f", "RED": "#e74c3c"},
    edgecolor="black",
    ax=ax,
)

ax.set_xlabel("Churn Risk Score")
ax.set_ylabel("Member Count")
ax.set_title("Distribution of Member Risk Scores")
st.pyplot(fig)

# --- Boxplot by risk band ---
st.subheader("Score Distribution by Band")

fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.boxplot(
    data=df,
    x="band",
    y="score",
    order=["GREEN", "YELLOW", "RED"],
    palette={"GREEN": "#2ecc71", "YELLOW": "#f1c40f", "RED": "#e74c3c"},
    ax=ax2,
)
ax2.set_xlabel("Risk Band")
ax2.set_ylabel("Risk Score")
ax2.set_title("Score Spread by Risk Category")
st.pyplot(fig2)
# --- Top 10 members at highest risk ---
st.subheader("🚨 Top 10 Members at Highest Risk")

top10 = df.sort_values("score", ascending=False).head(10)

# select key columns to display
st.dataframe(top10[["member_id", "score", "band", "days_since_last_checkin", "momentum_drop"]])
