# src/cf_buckhead_analytics/churn_regression.py
"""
Train churn classification models and surface the strongest performer.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .training_data import make_training_frame

FEATURE_COLUMNS: Sequence[str] = (
    "days_since_last_checkin",
    "momentum_drop",
    "purchase_drop",
    "pending_cancel_flag",
    "membership_age_days",
    "attendance_streak",
)


@dataclass
class ModelPerformance:
    """Container describing a single model's evaluation results."""

    name: str
    estimator: Any
    accuracy: float
    auc: float
    confusion: pd.DataFrame
    feature_stats: pd.DataFrame


@dataclass
class RegressionReport:
    """Summary of the best-performing model and a leaderboard of candidates."""

    best_model: ModelPerformance
    leaderboard: pd.DataFrame
    sample_size: int
    positive_rate: float


def _build_model_registry() -> dict[str, Any]:
    """Create model definitions to evaluate."""
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        C=1.5,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=42, learning_rate=0.1, n_estimators=200, max_depth=3
        ),
        "random_forest": RandomForestClassifier(
            random_state=42, n_estimators=400, max_depth=6, min_samples_leaf=5
        ),
    }


def _feature_importance_frame(estimator: Any, model_name: str) -> pd.DataFrame:
    """Extract feature importance style statistics for a fitted estimator."""
    feature_names = list(FEATURE_COLUMNS)

    if model_name == "logistic_regression":
        logistic = estimator.named_steps["clf"]
        coefficients = pd.Series(logistic.coef_[0], index=feature_names)
        ordered = coefficients.sort_values(ascending=False).to_frame(name="coefficient")
        return pd.DataFrame(ordered)

    if hasattr(estimator, "feature_importances_"):
        importances = pd.Series(estimator.feature_importances_, index=feature_names)
        ordered = importances.sort_values(ascending=False).to_frame(name="importance")
        return pd.DataFrame(ordered)

    # Fallback: empty dataframe when stats are not available
    return pd.DataFrame(columns=["feature", "value"])


def train_regression_model(as_of: str | None = None) -> RegressionReport:
    """
    Train several classifiers, compare accuracy/AUC, and return the top performer.

    This approach typically yields stronger metrics than a single unscaled logistic
    regression because it (a) balances classes, (b) scales numeric inputs, and
    (c) explores boosted-tree ensembles that capture non-linear effects.
    """
    df = make_training_frame(as_of=as_of)

    X = df[list(FEATURE_COLUMNS)]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model_registry = _build_model_registry()
    evaluations: list[ModelPerformance] = []

    for name, estimator in model_registry.items():
        fitted = estimator.fit(X_train, y_train)
        y_pred = fitted.predict(X_test)
        y_proba = fitted.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=pd.Index([0, 1], name="Actual"),
            columns=pd.Index([0, 1], name="Predicted"),
        )

        feature_stats = _feature_importance_frame(fitted, name)
        evaluations.append(
            ModelPerformance(
                name=name,
                estimator=fitted,
                accuracy=acc,
                auc=auc,
                confusion=cm_df,
                feature_stats=feature_stats,
            )
        )

    leaderboard = pd.DataFrame(
        {
            "model": [result.name for result in evaluations],
            "accuracy": [result.accuracy for result in evaluations],
            "auc": [result.auc for result in evaluations],
        }
    ).sort_values(by=["auc", "accuracy"], ascending=False, ignore_index=True)

    best_name = leaderboard.loc[0, "model"]
    best_result = next(result for result in evaluations if result.name == best_name)

    return RegressionReport(
        best_model=best_result,
        leaderboard=leaderboard,
        sample_size=len(df),
        positive_rate=float(y.mean()),
    )


if __name__ == "__main__":
    report = train_regression_model()
    best = report.best_model

    print("Best Model:", best.name)
    print(f"Sample size: {report.sample_size}")
    print(f"Positive rate: {report.positive_rate:.3f}")
    print(f"Accuracy: {best.accuracy:.3f}")
    print(f"AUC: {best.auc:.3f}")
    print("Confusion Matrix:\n", best.confusion)
    print("\nFeature Statistics:")
    print(best.feature_stats)
    print("\nLeaderboard:")
    print(report.leaderboard)
