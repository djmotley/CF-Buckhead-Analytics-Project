from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd


def load_sample_data(raw_dir: Path = Path("data/raw_sample_dylan")) -> dict[str, pd.DataFrame]:
    """
    Load Dylan's raw sample files from the specified directory and normalize them
    into a consistent schema. Returns a dictionary of pandas DataFrames with keys:
      - 'attendance': columns -> [member_id, date]
      - 'memberships': columns -> [member_id, canceled_on, pending_cancel]
      - 'members': columns -> [member_id, joined_on]
      - 'transactions': columns -> [member_id, date, amount]
    The function handles alternate column name spellings and ensures date fields
    are converted to datetime objects and amounts to numeric values.
    """
    # Verify the directory exists
    if not raw_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {raw_dir}")

    # Define file paths
    attendance_file = (
        raw_dir / "Attendance.csv"
    )  # Note: filename as given (possible typo "Attendence")
    cancellations_file = raw_dir / "Cancellations.csv"
    members_file = raw_dir / "Members.csv"
    sales_file = raw_dir / "Store_Sales.csv"

    # Check that each expected file exists before proceeding
    for file_path in [attendance_file, cancellations_file, members_file, sales_file]:
        if not file_path.exists():
            raise FileNotFoundError(f"Expected file not found: {file_path}")

    # Define synonyms for column names in each file to map to internal schema
    attendance_synonyms = {
        "member_id": ["Member_ID", "MemberID", "member_id", "memberID"],
        "date": ["Date", "date", "Class_Date", "AttendanceDate"],  # include plausible variants
    }
    cancellations_synonyms = {
        "member_id": ["Member_ID", "MemberID", "member_id", "memberID"],
        "canceled_on": [
            "Cancel_Date",
            "CancelDate",
            "Cancelled_Date",
            "Cancelled_On",
            "Canceled_On",
        ],
        # pending_cancel is not in raw data; we'll add it manually as False for all.
    }
    members_synonyms = {
        "member_id": ["Member_ID", "MemberID", "member_id", "memberID"],
        "joined_on": ["Join_Date", "JoinDate", "Joined_On", "JoinedOn", "join_date"],
        # We ignore First_Name, Membership_Type, Referral_Source for normalization.
    }
    transactions_synonyms = {
        "member_id": ["Member_ID", "MemberID", "member_id", "memberID"],
        "date": ["Purchase_Date", "PurchaseDate", "Date", "date"],
        "amount": ["Amount_USD", "Amount", "amount", "Sales_Amount", "Price"],
        # 'Item' column is ignored in the normalized transactions schema.
    }

    def _read_and_standardize(
        file_path: Path, synonyms_map: Mapping[str, Sequence[str]], required_cols: Sequence[str]
    ) -> pd.DataFrame:
        """Helper to read a CSV file and rename/select columns based on the provided synonyms map."""
        # Read the CSV into a DataFrame
        df = pd.read_csv(file_path)
        # Build a rename map for columns found in the DataFrame
        rename_map: dict[str, str] = {}
        for std_col, name_variants in synonyms_map.items():
            for col_name in name_variants:
                if col_name in df.columns:
                    rename_map[col_name] = std_col
                    break  # stop at the first match for this standard column
        # Rename columns in the DataFrame
        df = df.rename(columns=rename_map)
        # Verify that all required columns are now present after renaming
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found in {file_path.name}. "
                    f"Available columns: {list(df.columns)}"
                )
        # Keep only the required columns (drop others not needed for normalization)
        df = df[required_cols].copy()
        return df

    # Load and normalize each DataFrame
    attendance_df = _read_and_standardize(
        attendance_file, attendance_synonyms, required_cols=["member_id", "date"]
    )
    memberships_df = _read_and_standardize(
        cancellations_file, cancellations_synonyms, required_cols=["member_id", "canceled_on"]
    )
    members_df = _read_and_standardize(
        members_file, members_synonyms, required_cols=["member_id", "joined_on"]
    )
    transactions_df = _read_and_standardize(
        sales_file, transactions_synonyms, required_cols=["member_id", "date", "amount"]
    )

    # Add pending_cancel column to memberships (cancellations) – set to False for all rows
    memberships_df["pending_cancel"] = False  # boolean column

    # Convert date columns to datetime objects (NaT for invalid or missing dates)
    attendance_df["date"] = pd.to_datetime(attendance_df["date"], errors="coerce")
    memberships_df["canceled_on"] = pd.to_datetime(memberships_df["canceled_on"], errors="coerce")
    members_df["joined_on"] = pd.to_datetime(members_df["joined_on"], errors="coerce")
    transactions_df["date"] = pd.to_datetime(transactions_df["date"], errors="coerce")

    # Convert amount column to numeric (float). Invalid parsing will become NaN.
    transactions_df["amount"] = pd.to_numeric(transactions_df["amount"], errors="coerce")

    # Return a dictionary of the four normalized DataFrames
    return {
        "attendance": attendance_df,
        "memberships": memberships_df,
        "members": members_df,
        "transactions": transactions_df,
    }
