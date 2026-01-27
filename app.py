import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(__file__))

from cf_buckhead_analytics.dashboard import run_dashboard_app
import streamlit as st

st.set_page_config(page_title="CF Buckhead Analytics Dashboard")

if __name__ == "__main__":
    run_dashboard_app()
