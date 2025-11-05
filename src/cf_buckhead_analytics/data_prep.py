# src/cf_buckhead_analytics/data_prep.py
"""
Data ingestion helpers that transform raw PushPress-style exports into
normalized tables ready for feature engineering and modeling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import config


@dataclass(frozen=True)
class RawData:
    """Container for cleaned attendance plus optional member metadata."""

    attendance: pd.DataFrame
    members: pd.DataFrame | None


def load_raw_data(raw_dir: Path | None = None) -> RawData:
    """
    Load the raw attendance (mandatory) and member roster (optional) files.

    Parameters
    ----------
    raw_dir:
        Optional override for the raw directory. Defaults to config.RAW_DIR.
    """

    base_dir = Path(raw_dir) if raw_dir else Path(config.RAW_DIR)
    if not base_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {base_dir}")

    attendance_path = base_dir / "Attendance.csv"
    if not attendance_path.exists():
        raise FileNotFoundError(
            f"Required file Attendance.csv not found in {attendance_path.parent}"
        )

    attendance = _load_attendance(attendance_path)

    members_path = base_dir / "Members.csv"
    members = _load_members(members_path, attendance) if members_path.exists() else None

    return RawData(attendance=attendance, members=members)


def ensure_directories() -> None:
    """Create the downstream directories if they do not already exist."""

    for path in (
        config.PROCESSED_DIR,
        config.REPORTS_OUTREACH_DIR,
        config.MODELS_DIR,
    ):
        Path(path).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Internal helpers


def _to_snake_case(name: str) -> str:
    return (
        name.replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("__", "_")
        .strip()
        .lower()
    )


def _load_attendance(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    rename_map: dict[str, str] = {}
    for col in df.columns:
        snake = _to_snake_case(col)
        if snake in {"member_id", "class_ts", "class_type"}:
            rename_map[col] = snake
        elif snake in {"memberid", "member"}:
            rename_map[col] = "member_id"
        elif snake in {"datetime", "checkin_time", "check_in_time"}:
            rename_map[col] = "class_ts"
    df = df.rename(columns=rename_map)

    required = {"member_id", "class_ts"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Attendance file is missing required columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    df["member_id"] = df["member_id"].astype(str)
    df["class_ts"] = pd.to_datetime(df["class_ts"], errors="coerce")
    if df["class_ts"].isna().any():
        df = df.dropna(subset=["class_ts"]).copy()

    today = pd.Timestamp.today().normalize()
    future_mask = df["class_ts"].dt.normalize() > today
    if future_mask.any():
        # shift future check-ins back to the reference day while preserving time of day
        delta = df.loc[future_mask, "class_ts"].dt.normalize() - today
        df.loc[future_mask, "class_ts"] = df.loc[future_mask, "class_ts"] - delta

    if "class_type" not in df.columns:
        df["class_type"] = "CrossFit"
    df["class_type"] = df["class_type"].fillna("CrossFit")

    df = df.sort_values(["member_id", "class_ts"]).reset_index(drop=True)
    return df[["member_id", "class_ts", "class_type"]]


def _load_members(path: Path, attendance: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path)
    columns: pd.Index = pd.Index([_to_snake_case(col) for col in df.columns])
    df.columns = columns

    column_aliases: dict[str, Iterable[str]] = {
        "member_id": ("member_id", "memberid", "id"),
        "status": ("status", "membership_status"),
        "plan": ("plan", "plan_name", "membership_type"),
        "member_since": ("member_since", "membersince", "first_checkin", "first_seen"),
        "plan_cancel_date": ("plan_cancel_date", "cancel_date", "plan_end_date"),
        "plan_end_date": ("plan_end_date", "membership_end_date", "end_date"),
    }

    for target, candidates in column_aliases.items():
        if target in df.columns:
            continue
        for candidate in candidates:
            if candidate in df.columns:
                df = df.rename(columns={candidate: target})
                break

    if "member_id" not in df.columns:
        raise ValueError("Members.csv must contain a member_id column.")

    df["member_id"] = df["member_id"].astype(str)

    for date_col in ("member_since", "plan_cancel_date", "plan_end_date"):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        else:
            df[date_col] = pd.NaT

    attendance_first_seen = (
        attendance.groupby("member_id")["class_ts"].min().rename("attendance_first_seen")
    )
    df = df.merge(attendance_first_seen, on="member_id", how="left")
    df["member_since"] = df["member_since"].fillna(df["attendance_first_seen"])
    df = df.drop(columns=["attendance_first_seen"])

    missing_member_since = df["member_since"].isna()
    if missing_member_since.any():
        df.loc[missing_member_since, "member_since"] = df.loc[
            missing_member_since, "plan_cancel_date"
        ]

    end_missing = df["plan_end_date"].isna() & df["plan_cancel_date"].notna()
    df.loc[end_missing, "plan_end_date"] = df.loc[end_missing, "plan_cancel_date"]

    if "status" in df.columns:
        status_series = df["status"]
    else:
        status_series = pd.Series("", index=df.index, dtype="object")
    df["status"] = status_series.fillna("").astype(str)

    if "plan" in df.columns:
        plan_series = df["plan"]
    else:
        plan_series = pd.Series("", index=df.index, dtype="object")
    df["plan"] = plan_series.fillna("").astype(str)

    df = df.sort_values(
        ["member_id", "plan_end_date", "plan_cancel_date", "member_since"],
        ascending=[True, True, True, True],
    ).drop_duplicates("member_id", keep="last")

    return df[
        [
            "member_id",
            "status",
            "plan",
            "member_since",
            "plan_cancel_date",
            "plan_end_date",
        ]
    ]
