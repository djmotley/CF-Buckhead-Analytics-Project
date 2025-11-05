import sys
from pathlib import Path

# Ensure we can import the package from src/
repo_root = Path(__file__).resolve().parent
sys.path.append(str(repo_root / "src"))

from cf_buckhead_analytics.dashboard import run_dashboard_app

if __name__ == "__main__":
    run_dashboard_app()
