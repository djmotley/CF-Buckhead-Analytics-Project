from __future__ import annotations

from cf_buckhead_analytics.training_data import make_training_frame


def test_training_data_positive_rate_bounds() -> None:
    artifacts = make_training_frame()
    df = artifacts.dataset

    assert "churned" in df.columns
    assert set(df["churned"].unique()).issubset({0, 1})
    assert 0.0 <= artifacts.positive_rate <= 1.0

    # Ensure label metadata columns exist
    for column in ("churn_event_type", "snapshot_as_of"):
        assert column in df.columns
