from __future__ import annotations

from cf_buckhead_analytics.feature_engineering import generate_feature_store


def test_feature_store_contains_required_columns() -> None:
    artifacts = generate_feature_store()
    df = artifacts.features

    expected_columns = {
        "member_id",
        "first_seen",
        "last_checkin",
        "tenure_days",
        "days_since_last_checkin",
        "attend_recent_28",
        "attend_prior_29_84",
        "lifetime_attend",
        "class_type_share_endurance",
        "tenure_bucket",
        "snapshot_as_of",
    }

    assert not df.empty
    assert expected_columns.issubset(df.columns)
    assert (df["days_since_last_checkin"] >= 0).all()
