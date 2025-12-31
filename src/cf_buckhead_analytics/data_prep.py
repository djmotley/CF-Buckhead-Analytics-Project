# src/cf_buckhead_analytics/data_prep.py
"""
Data ingestion helpers that transform raw PushPress-style exports into
normalized tables ready for feature engineering and modeling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import config


@dataclass(frozen=True)
class RawData:
    """Container for cleaned attendance plus optional membership metadata."""

    attendance: pd.DataFrame
    memberships: pd.DataFrame | None


def normalize_plan(plan: str | float | None) -> str:
    """Map free-form plan names into standard buckets."""
    if plan is None or (isinstance(plan, float) and pd.isna(plan)):
        return "Other"
    text = str(plan).strip().lower()
    if not text or text in {"nan", "none"}:
        return "Other"

    if text in {"coach", "staff", "vip staff"}:
        return "Coach/Staff"

    if "unlimited" in text or "$149/month: unlimited" in text or "$189/month: unlimited" in text:
        return "Unlimited"
    if "vip" in text and "coach" not in text:
        return "Unlimited"

    limited_tokens = [
        "/month",
        "x/month",
        "classes / month",
        "5x",
        "6x",
        "8x",
        "12x",
    ]
    if any(token in text for token in limited_tokens) or re.search(r"\b\d+\s*x\b", text):
        return "Limited"

    return "Other"


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
    memberships = _load_memberships(members_path, attendance) if members_path.exists() else None

    return RawData(attendance=attendance, memberships=memberships)


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


def _load_memberships(path: Path, attendance: pd.DataFrame) -> pd.DataFrame:
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
        "first_name": ("first_name", "firstname", "first"),
        "last_name": ("last_name", "lastname", "last"),
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

    member_since_raw = df.get("member_since")
    if member_since_raw is not None:
        df["member_since"] = pd.to_datetime(member_since_raw, errors="coerce")
    else:
        df["member_since"] = pd.NaT

    plan_cancel_raw = df.get("plan_cancel_date")
    if plan_cancel_raw is None:
        plan_cancel_raw = df.get("plancanceldate")
    if plan_cancel_raw is not None:
        df["plan_cancel_date"] = pd.to_datetime(plan_cancel_raw, errors="coerce")
    else:
        df["plan_cancel_date"] = pd.NaT

    plan_end_raw = df.get("plan_end_date")
    if plan_end_raw is None:
        plan_end_raw = df.get("planenddate")
    if plan_end_raw is not None:
        df["plan_end_date"] = pd.to_datetime(plan_end_raw, errors="coerce")
    else:
        df["plan_end_date"] = pd.NaT

    status_raw = df.get("status")
    if status_raw is not None:
        status_series = status_raw.fillna("").astype(str).str.lower()
    else:
        status_series = pd.Series("", index=df.index, dtype="object")
    df["status"] = status_series

    plan_raw = df.get("plan")
    if plan_raw is not None:
        plan_series = plan_raw.fillna("").astype(str)
    else:
        plan_series = pd.Series("", index=df.index, dtype="object")
    df["plan"] = plan_series
    df["plan_norm"] = plan_series.apply(normalize_plan)
    df["first_name"] = df.get("first_name", pd.Series("", index=df.index, dtype="object")).fillna(
        ""
    )
    df["last_name"] = df.get("last_name", pd.Series("", index=df.index, dtype="object")).fillna("")

    attendance_first_seen = (
        attendance.groupby("member_id")["class_ts"].min().rename("attendance_first_seen")
    )
    df = df.merge(attendance_first_seen, on="member_id", how="left")
    df["member_since"] = df["member_since"].fillna(df["attendance_first_seen"])
    df = df.drop(columns=["attendance_first_seen"])

    df = df.sort_values(
        ["member_id", "plan_end_date", "plan_cancel_date", "member_since"],
        ascending=[True, True, True, True],
    ).drop_duplicates("member_id", keep="last")

    return df[
        [
            "member_id",
            "status",
            "first_name",
            "last_name",
            "plan",
            "plan_norm",
            "plan_cancel_date",
            "plan_end_date",
            "member_since",
        ]
    ]
