# Fighting Member Churn with Data

**An Aspiring Analyst’s Journey to Solving Real Business Problems at CrossFit Buckhead**

🔴 **Live Dashboard:**
[https://cf-buckhead-analytics-project.streamlit.app/](https://cf-buckhead-analytics-project.streamlit.app/)

---

## Why I Started

For a long time, I struggled to find a clear starting point for my career. I wanted to commit to something I genuinely enjoyed doing, and that ultimately led me to business analytics. This project began as a personal litmus test to answer one honest question:
**Do I actually enjoy the work that business analysts do?**

Rather than guessing, I decided to try to replicate the work real business analysts do. I chose a real-world business problem with real constraints and real ambiguity. The result was one of the most challenging and rewarding learning experiences I’ve had, which helped confirm that this is the career path I want to pursue.

In addition to self-discovery, I wanted to find a way to give back to CrossFit Buckhead, a community that has given me so much since my move to Atlanta. Although the result was not as earth-shattering as I wanted, much insight was gained, and many thought-provoking questions were raised.

---

## Defining the Problem

The business I chose was **CrossFit Buckhead**, a gym I’m a member of and previously coached at. After meeting with the owner and listening to her pain points, one issue stood out clearly:
**Member churn.**

For a small, membership-based business, churn has an outsized impact. Acquiring new members is expensive, while retaining existing ones is far more sustainable. This project aimed to answer a core question:
**Can data help us understand why members leave and how to retain them more effectively?**

---

## The Challenge and My Starting Point

Starting this project was overwhelming. I was balancing GMAT preparation, work, training, and volunteering—all while learning an entirely new technical domain. With guidance from a mentor and AI-assisted tools, I established a foundation and iterated from there.

### Leveraging AI as a Learning Accelerator
AI tools like GitHub Copilot and ChatGPT Codex played a significant role in:
- Drafting notebook structures
- Debugging errors and edge cases
- Understanding analytics workflows

This approach reflects how I see my future role as a business analyst:
**Bridging data, technology, and business strategy.**

---

## Building the Project

### 1. Data Discovery
I began by researching churn predictors in subscription-based businesses and identifying meaningful features like attendance frequency, streaks, and engagement gaps. CrossFit Buckhead’s data came from **PushPress**, a gym management platform.

### 2. Data Understanding and Cleaning
Working with real PushPress exports taught me that operational data is rarely clean. I gained hands-on experience with:
- Messy, imperfect data
- Feature engineering
- Translating raw records into behavior-based metrics

### 3. Modeling and Interpretation
Given the small dataset and the need for interpretability, I focused on **gradient-boosted decision trees**. This choice prioritized:
- Clear explanations of feature impact
- Actionable insights over marginal performance gains

### 4. Turning Insights into Action (Dashboard)
I built a **Streamlit dashboard** to visualize engagement patterns and churn-related trends. This taught me how to:
- Connect data pipelines to a front-end tool
- Think critically about usability
- Translate analysis into something tangible

---

## Key Learnings and Reflections

This project taught me far more than technical skills:
- **Asking the right questions** matters more than writing perfect code.
- Simpler, interpretable models often provide more business value.
- Real data introduces problems that cannot be ignored.
- Tools (including AI) are most powerful when used thoughtfully.

Most importantly, I discovered that I enjoy the ambiguity, problem-solving, and communication inherent in business analytics.

---

## Impact, Limitations, and Next Steps

### Achievements
- A complete project structure
- A working churn model
- A live dashboard with clear, actionable insights

### Improvements Moving Forward
- Refining churn definitions and labeling logic
- Improving feature engineering (e.g., tenure normalization, engagement decay)
- Optimizing dashboard performance and data pipelines
- Adding clearer, business-oriented visualizations

---

## Technical Snapshot

- **Tools:** Python, Pandas, Scikit-learn, Streamlit, Jupyter Notebooks, GitHub
- **Modeling:** Gradient Boosted Trees (XGBoost/LightGBM/sklearn)
- **Data Source:** Real PushPress exports (attendance and membership data)
- **Objective:** Understand and predict member churn; visualize engagement trends

### Project Structure
```
CF-Buckhead-Analytics-Project/
├── .vscode/                                     # Editor configuration
├── data/                                        # Project datasets (raw, cleaned, and intermediate)
├── models/                                      # Saved models and model artifacts
├── reports/
│   └── outreach/                                # Generated outreach lists and reporting outputs
├── src/
│   └── cf_buckhead_analytics/                   # Core analytics package
│       ├── __init__.py
│       ├── data_prep.py                         # Data ingestion and cleaning logic
│       ├── feature_engineering.py               # Feature engineering pipelines
│       ├── training_data.py                     # Training dataset creation and labeling
│       ├── churn_regression.py                  # Churn model training and evaluation
│       └── utils.py                             # Shared helper functions
├── tests/                                       # Unit and integration tests
├── app.py                                       # Streamlit application entry point
├── Project Introduction Meeting_CFB-AP.pdf      # Project background and stakeholder context
├── README.md                                    # Project overview and documentation
├── requirements.txt                             # Python dependencies
├── pyproject.toml                               # Project configuration and tooling
├── Makefile                                     # Common development commands
├── .pre-commit-config.yaml                      # Code quality and formatting hooks
├── .editorconfig                                # Editor formatting standards
├── .gitattributes
└── .gitignore
```

---

## Acknowledgments

- **Dylan Alexander:** For mentorship, technical guidance, and patience
- **Alison Giannavola (CrossFit Buckhead Owner):** For trust, access, and support
- **Carl Gold:** For inspiration and insights on churn analytics

---

## How to Reproduce

1. Clone the repository:
   ```bash
   git clone https://github.com/djmotley/CF-Buckhead-Analytics-Project.git
   cd CF-Buckhead-Analytics-Project
   git checkout Final-Branch
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Run the pipeline:
   ```bash
   python -m cf_buckhead_analytics.data_prep
   python -m cf_buckhead_analytics.feature_engineering
   python -m cf_buckhead_analytics.training_data
   python -m cf_buckhead_analytics.churn_regression
   ```

4. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```

---

This project is a living document, reflecting my growth as an aspiring business analyst. Every iteration represents a new skill learned, a limitation discovered, or a better question asked.
