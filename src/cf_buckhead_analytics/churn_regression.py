# src/cf_buckhead_analytics/churn_regression.py
"""
Train multiple churn prediction models and export outreach-ready scores.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from .data_prep import load_raw_data
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
    model_paths: dict[str, Path]
    metadata_paths: dict[str, Path]
    feature_importance_paths: dict[str, Path]
    outreach_path: Path | None


def _standardize_plan(plan: Any) -> str:
    """Standardize plan names into predefined categories."""
    if plan is None or (isinstance(plan, float) and pd.isna(plan)):
        return "Other"
    text = str(plan).strip().lower()
    if not text:
        return "Other"
    if any(keyword in text for keyword in ("coach", "staff", "owner", "comp")):
        return "Coach/Staff"
    if "unlimited" in text:
        return "Unlimited"
    if re.search(r"\b\d+\s*(x|visit|visits|class|classes|session|sessions)\b", text):
        return "Limited"
    return "Other"


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def _get_feature_mapping(preprocessor: ColumnTransformer) -> tuple[list[str], list[str]]:
    feature_names: list[str] = []
    base_features: list[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if transformer == "drop":
            continue
        if name == "num":
            for col in cols:
                feature_names.append(str(col))
                base_features.append(str(col))
        elif name == "cat":
            ohe = preprocessor.named_transformers_.get(name)
            if hasattr(ohe, "categories_"):
                for col, categories in zip(cols, ohe.categories_):
                    for category in categories:
                        feature_names.append(f"{col}_{category}")
                        base_features.append(str(col))
            else:
                for col in cols:
                    feature_names.append(str(col))
                    base_features.append(str(col))
        else:
            for col in cols:
                feature_names.append(str(col))
                base_features.append(str(col))
    return feature_names, base_features


def _aggregate_importances(
    preprocessor: ColumnTransformer, importances: np.ndarray
) -> pd.DataFrame:
    feature_names, base_features = _get_feature_mapping(preprocessor)
    flat_importances = np.asarray(importances).ravel()
    usable = min(len(feature_names), len(flat_importances))
    if usable == 0:
        return pd.DataFrame(columns=["feature", "importance", "importance_std"])
    data = pd.DataFrame(
        {
            "feature": base_features[:usable],
            "importance": flat_importances[:usable],
        }
    )
    # Use agg to ensure a DataFrame result before sorting (avoids Series.sort_values 'by' overload issues)
    agg = (
        data.groupby("feature", as_index=False)
        .agg({"importance": "sum"})
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )
    agg["importance_std"] = 0.0
    return agg.astype({"feature": str, "importance": float, "importance_std": float})


def _evaluate_predictions(y_true: pd.Series, y_scores: np.ndarray) -> dict[str, float]:
    pr_auc = average_precision_score(y_true, y_scores)
    share = config.CHURN_TOP_SHARE
    top_k = max(int(len(y_scores) * share), 1)
    cutoff = np.sort(y_scores)[-top_k] if top_k < len(y_scores) else float(np.min(y_scores))
    predicted_positive = y_scores >= cutoff

    positives = predicted_positive.sum()
    true_positive_mask = (y_true == 1) & predicted_positive
    true_positive_count = int(true_positive_mask.sum())
    actual_positive_count = int((y_true == 1).sum())

    precision_at_share = true_positive_count / positives if positives else 0.0
    recall_at_share = true_positive_count / actual_positive_count if actual_positive_count else 0.0
    return {
        "pr_auc": float(pr_auc),
        "precision_at_top_share": float(precision_at_share),
        "recall_at_top_share": float(recall_at_share),
        "cutoff": float(cutoff),
    }


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

    # Preprocess dataset
    dataset = dataset[dataset["plan_norm"] != "Coach/Staff"]
    if dataset.empty:
        raise ValueError("No eligible records remain after filtering plan_norm == 'Coach'.")

    X = dataset[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = dataset["churned"].astype(int)

    print(f"Training samples: {len(dataset)} (positive rate {float(y.mean()):.3f})")

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=config.RANDOM_STATE
    )

    # Train gradient-boosted decision tree model
    preprocessor = _build_preprocessor()
    model = HistGradientBoostingClassifier(random_state=config.RANDOM_STATE)
    pipeline = Pipeline([("prep", preprocessor), ("model", model)])
    pipeline.fit(x_train, y_train)

    y_scores = pipeline.predict_proba(x_test)[:, 1]
    metrics = _evaluate_predictions(y_test, y_scores)

    # Save model and metadata
    model_dir = Path(config.MODELS_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"churn_model_{artifacts.as_of.date()}_histgb.pkl"
    joblib.dump(pipeline, model_path)

    metadata = {
        "model_type": "HistGradientBoosting",
        "trained_on": artifacts.as_of.date().isoformat(),
        "n_samples": len(dataset),
        "positive_rate": float(y.mean()),
        "pr_auc": metrics["pr_auc"],
        "precision_at_top_share": metrics["precision_at_top_share"],
        "recall_at_top_share": metrics["recall_at_top_share"],
    }
    metadata_path = model_dir / f"model_metadata_{artifacts.as_of.date()}_histgb.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Model trained and saved: {model_path}")
    print(f"Metadata saved: {metadata_path}")

    return artifacts, ModelOutputs(
        model_paths={"histgb": model_path},
        metadata_paths={"histgb": metadata_path},
        feature_importance_paths={},
        outreach_path=None,
    )


def _write_outreach_scores(pipeline: Pipeline, dataset: pd.DataFrame, as_of: pd.Timestamp) -> Path:
    scores = pipeline.predict_proba(dataset[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    required_cols = [
        "member_id",
        "first_name",
        "last_name",
        "days_since_last_checkin",
        "attend_recent_28",
        "tenure_bucket",
        "plan_norm",
    ]
    available_cols = [col for col in required_cols if col in dataset.columns]
    result = dataset[available_cols].copy()
    result["score"] = scores

    result["plan_norm"] = result["plan_norm"].apply(_standardize_plan)
    result = result[result["plan_norm"].isin(["Unlimited", "Limited"])]

    result["days_since_last_checkin"] = pd.to_numeric(
        result["days_since_last_checkin"], errors="coerce"
    )
    result = result[result["days_since_last_checkin"].between(21, 45, inclusive="both")].dropna(
        subset=["days_since_last_checkin"]
    )

    result = result.sort_values("score", ascending=False).head(config.OUTREACH_MAX_MEMBERS)

    first_names = (
        (
            result["first_name"]
            if "first_name" in result.columns
            else pd.Series("", index=result.index)
        )
        .astype(str)
        .str.strip()
    )
    last_names = (
        (
            result["last_name"]
            if "last_name" in result.columns
            else pd.Series("", index=result.index)
        )
        .astype(str)
        .str.strip()
    )
    member_names = (first_names + " " + last_names).str.strip()
    missing_name_mask = member_names.eq("")

    if missing_name_mask.any():
        try:
            raw = load_raw_data()
        except Exception:
            raw = None
        if raw and raw.memberships is not None:
            name_lookup = raw.memberships[
                [
                    col
                    for col in ["member_id", "first_name", "last_name"]
                    if col in raw.memberships.columns
                ]
            ].drop_duplicates("member_id")
            name_lookup["first_name"] = (
                name_lookup.get("first_name", pd.Series("", index=raw.memberships.index))
                .fillna("")
                .astype(str)
                .str.strip()
            )
            name_lookup["last_name"] = (
                name_lookup.get("last_name", pd.Series("", index=raw.memberships.index))
                .fillna("")
                .astype(str)
                .str.strip()
            )
            name_lookup["member_name"] = (
                name_lookup["first_name"] + " " + name_lookup["last_name"]
            ).str.strip()
            result = result.merge(
                name_lookup[["member_id", "member_name"]],
                on="member_id",
                how="left",
                suffixes=("", "_lookup"),
            )
            member_names = pd.Series(
                np.where(
                    member_names.eq(""),
                    result["member_name_lookup"].fillna(""),
                    member_names,
                ),
                index=result.index,
            )
            result = result.drop(columns=["member_name_lookup"], errors="ignore")

    result["member_name"] = pd.Series(
        np.where(member_names.eq(""), result["member_id"], member_names),
        index=result.index,
    )

    result["risk_tier"] = pd.cut(
        result["score"],
        bins=[-np.inf, 0.3, 0.5, 0.7, np.inf],
        labels=["Low", "Medium", "High", "Very High"],
    ).astype(str)
    result["risk_score"] = result["score"].clip(0, 1)

    keep_columns = [
        "member_id",
        "member_name",
        "plan_norm",
        "tenure_bucket",
        "days_since_last_checkin",
        "attend_recent_28",
        "risk_score",
        "risk_tier",
    ]
    result = result[keep_columns]
    result["risk_score"] = result["risk_score"].map(lambda x: f"{x:.2f}")

    outreach_path = Path(config.REPORTS_OUTREACH_DIR) / f"outreach_{as_of.date()}.csv"
    Path(config.REPORTS_OUTREACH_DIR).mkdir(parents=True, exist_ok=True)
    result.to_csv(outreach_path, index=False)
    return outreach_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train churn models and export outreach files.")
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
