# CrossFit Buckhead Retention Analytics

Using data to proactively identify at-risk members and improve retention.

---

## 📖 Overview

This project delivers a complete analytics pipeline and interactive dashboard that helps **CrossFit Buckhead** understand and reduce member churn. The system consolidates attendance, cancellation, membership, and retail purchase data to:
- Spot early signs of disengagement
- Generate member-level churn risk scores
- Surface high-priority outreach opportunities

The end-to-end workflow transforms raw inputs into cleaned datasets, engineered features, churn risk scores, and a dashboard you can explore with Streamlit.

---

## ⚙️ Project Structure

```text
crossfit-buckhead-analytics/
├── data/
│   ├── raw_sample_dylan/        # sample PushPress-style exports
│   └── processed/               # generated churn score files
├── notebooks/                   # exploratory analysis and prototyping
├── reports/
│   ├── outreach/                # top-10 high-risk member reports
│   └── figures/                 # saved visuals (screenshots, etc.)
├── src/
│   └── cf_buckhead_analytics/
│       ├── config.py            # global settings and file paths
│       ├── data_prep.py         # loads & cleans raw data
│       ├── feature_engineering.py # builds analytic features
│       ├── risk_scoring.py      # computes churn risk scores
│       ├── dashboard.py         # Streamlit dashboard interface
│       └── __init__.py
├── tests/                       # unit tests for key modules
├── pyproject.toml
└── README.md
```

---

## 🚀 Pipeline Workflow

1. **Data Preparation (`data_prep.py`)**
   Loads CSV exports, standardizes formats, and creates unified member views.
2. **Feature Engineering (`feature_engineering.py`)**
   Builds behavioral metrics such as days since last check-in, attendance momentum, retail cadence, and cancellation flags.
3. **Risk Scoring (`risk_scoring.py`)**
   Applies a weighted formula to produce churn risk scores (0–100) and tags members:
   - 🟢 **Green:** Low risk
   - 🟡 **Yellow:** Moderate risk
   - 🔴 **Red:** High risk
   Outputs are written to `data/processed/` and outreach lists to `reports/outreach/`.
4. **Dashboard (`dashboard.py`)**
   Streamlit app that summarizes the latest results with metrics, histograms, boxplots, and top at-risk members. Users can rerun the latest analysis and browse historical score files.

---

## 🛠️ Getting Started

1. Install dependencies (`pip install -e .` from the project root).
2. Place new data exports inside `data/raw_sample_dylan/` (or adjust paths in `config.py`).
3. Run the pipeline scripts in order:
   ```bash
   python src/cf_buckhead_analytics/data_prep.py
   python src/cf_buckhead_analytics/feature_engineering.py
   python src/cf_buckhead_analytics/risk_scoring.py
   ```

---

## 📊 Run the Dashboard

Launch the Streamlit app to explore the latest outputs:

```bash
streamlit run src/cf_buckhead_analytics/dashboard.py
```

Open your browser to `http://localhost:8501` to view live metrics, charts, and outreach tables powered by the data in `data/processed/`.

---

## 🔍 Key Insights Demonstrated

- Data engineering with Pandas and modular Python scripts
- Feature design for predictive analytics use cases
- Weight-based churn scoring tailored to boutique fitness
- Visualization with Seaborn and Matplotlib
- Web app deployment via Streamlit
- Clean project organization and version control with Git/GitHub

---

## 🧩 Future Enhancements

- Integrate real-time PushPress API data
- Incorporate class attendance streaks or PR tracking as behavioral features
- Experiment with machine learning models for predictive scoring
- Automate weekly refreshes with cron jobs or GitHub Actions

---

## 👤 Authors

Dylan Alexander
Derek Motley
Business & Data Analytics Enthusiast · CrossFit Level 1 Trainer
Atlanta, GA · [LinkedIn](https://www.linkedin.com/in/derekmotley/)
