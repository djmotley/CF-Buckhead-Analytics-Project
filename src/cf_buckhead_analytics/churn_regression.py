# src/cf_buckhead_analytics/churn_regression.py
"""
Train the churn prediction model and export outreach-ready scores.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .training_data import TrainingArtifacts, make_training_frame, save_training_frame

NUMERIC_FEATURES = [
    "days_since_last_checkin",
    "attend_recent_28",
    "attend_prior_29_84",
    "lifetime_attend",
    "tenure_days",
]

CATEGORICAL_FEATURES = ["tenure_bucket", "plan_norm"]


@dataclass(frozen=True)
class ModelOutputs:
    model_path: Path
    metadata_path: Path
    feature_importance_path: Path
    outreach_path: Path


def train_gradient_boosted_model(
    as_of: str | None = None,
) -> tuple[TrainingArtifacts, ModelOutputs | None]:
    artifacts: TrainingArtifacts = make_training_frame(as_of=as_of)
    dataset = artifacts.dataset.copy()

    if dataset.empty or dataset["churned"].sum() == 0:
        save_training_frame(artifacts)
        print(
            "No positive churn labels for snapshot "
            f"{artifacts.as_of.date()}. Skipping model training."
        )
        return artifacts, None

    if "plan_norm" in dataset.columns:
        plan_norm_series = dataset["plan_norm"].astype(str)
    else:
        plan_norm_series = pd.Series("Unknown", index=dataset.index, dtype="object")
    dataset["plan_norm"] = plan_norm_series.fillna("Unknown").astype(str)
    dataset = dataset[dataset["plan_norm"].str.lower() != "coach"]
    if dataset.empty:
        raise ValueError("No eligible records remain after filtering plan_norm == 'Coach'.")

    if "tenure_bucket" in dataset.columns:
        tenure_bucket_series = dataset["tenure_bucket"].astype(str)
    else:
        tenure_bucket_series = pd.Series("Unknown", index=dataset.index, dtype="object")
    dataset["tenure_bucket"] = tenure_bucket_series.fillna("Unknown").astype(str)

    X = dataset[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = dataset["churned"].astype(int)

    print(f"Training samples: {len(dataset)} " f"(positive rate {float(y.mean()):.3f})")

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=config.RANDOM_STATE
    )

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    model = HistGradientBoostingClassifier(random_state=config.RANDOM_STATE)
    pipeline = Pipeline([("prep", preprocessor), ("model", model)])
    pipeline.fit(x_train, y_train)

    y_scores = pipeline.predict_proba(x_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_scores)

    share = config.CHURN_TOP_SHARE
    top_k = max(int(len(y_scores) * share), 1)
    cutoff = np.sort(y_scores)[-top_k] if top_k < len(y_scores) else np.min(y_scores)
    predicted_positive = y_scores >= cutoff

    precision_at_share = (y_test[predicted_positive] == 1).sum() / max(predicted_positive.sum(), 1)
    recall_at_share = (y_test[predicted_positive] == 1).sum() / max((y_test == 1).sum(), 1)

    if y_test.nunique() > 1 and (y_test == 1).any():
        perm_result = permutation_importance(
            pipeline,
            x_test,
            y_test,
            n_repeats=10,
            random_state=config.RANDOM_STATE,
            scoring="average_precision",
        )
        importance_df = pd.DataFrame(
            {
                "feature": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
                "importance": perm_result.importances_mean,
                "importance_std": perm_result.importances_std,
            }
        ).sort_values("importance", ascending=False)
    else:
        importance_df = pd.DataFrame(
            {
                "feature": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
                "importance": 0.0,
                "importance_std": 0.0,
            }
        )

    model_dir = Path(config.MODELS_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"churn_model_{artifacts.as_of.date()}.pkl"
    metadata_path = model_dir / f"model_metadata_{artifacts.as_of.date()}.json"
    feature_importance_path = (
        Path(config.PROCESSED_DIR) / f"feature_importance_{artifacts.as_of.date()}.csv"
    )

    joblib.dump(pipeline, model_path)
    importance_df.to_csv(feature_importance_path, index=False)

    metadata: dict[str, object] = {
        "as_of": artifacts.as_of.date().isoformat(),
        "n_samples": int(len(dataset)),
        "positive_rate": float(dataset["churned"].mean()),
        "pr_auc": float(pr_auc),
        "precision_at_top_share": float(precision_at_share),
        "recall_at_top_share": float(recall_at_share),
        "top_share": config.CHURN_TOP_SHARE,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    outreach_path = _write_outreach_scores(pipeline, dataset, artifacts.as_of)

    top_driver_rows = importance_df.head(5)
    if not top_driver_rows.empty:
        print("Top feature drivers:")
        for _, row in top_driver_rows.iterrows():
            print(f"  - {row['feature']}: {row['importance']:.4f}")

    print(
        f"PR-AUC: {pr_auc:.3f}; "
        f"Precision@Top {int(config.CHURN_TOP_SHARE * 100)}%: {precision_at_share:.3f}; "
        f"Recall@Top {int(config.CHURN_TOP_SHARE * 100)}%: {recall_at_share:.3f}; "
        f"Saved outreach to {outreach_path}"
    )

    save_training_frame(artifacts)  # persist the dataset used for modeling

    outputs = ModelOutputs(
        model_path=model_path,
        metadata_path=metadata_path,
        feature_importance_path=feature_importance_path,
        outreach_path=outreach_path,
    )
    return artifacts, outputs


def _write_outreach_scores(pipeline: Pipeline, dataset: pd.DataFrame, as_of: pd.Timestamp) -> Path:
    scores = pipeline.predict_proba(dataset[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    result = dataset[
        ["member_id", "days_since_last_checkin", "attend_recent_28", "tenure_bucket", "plan_norm"]
    ].copy()
    result["score"] = scores
    result = result.sort_values("score", ascending=False).head(config.OUTREACH_MAX_MEMBERS)
    result["risk_tier"] = pd.cut(
        result["score"],
        bins=[-np.inf, 0.3, 0.5, 0.7, np.inf],
        labels=["Low", "Medium", "High", "Very High"],
    ).astype(str)
    outreach_path = Path(config.REPORTS_OUTREACH_DIR) / f"outreach_{as_of.date()}.csv"
    Path(config.REPORTS_OUTREACH_DIR).mkdir(parents=True, exist_ok=True)
    result.to_csv(outreach_path, index=False)
    return outreach_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train churn model and export outreach files.")
    parser.add_argument(
        "--as_of",
        type=str,
        default=None,
        help="Optional snapshot date (YYYY-MM-DD). Defaults to latest available <= today.",
    )
    args = parser.parse_args(argv)
    train_gradient_boosted_model(as_of=args.as_of)


if __name__ == "__main__":
    main()
