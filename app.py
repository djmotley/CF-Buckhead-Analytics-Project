import sys
from pathlib import Path

# Add src to path for imports to work on Streamlit Cloud
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cf_buckhead_analytics.dashboard import run_dashboard_app

if __name__ == "__main__":
    run_dashboard_app()
