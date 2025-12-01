



"""
data_prep.py

Orchestrates data preparation for the SafePlayAI Discord phishing dataset.


"""

import os
import pandas as pd
from pathlib import Path


def load_raw_data(path: str) -> pd.DataFrame:
    print(f"[INFO] Loading raw data from: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Raw shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop totally empty / unnamed columns
    unnamed_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed_cols:
        print(f"[INFO] Dropping columns: {unnamed_cols}")
        df = df.drop(columns=unnamed_cols)

    # Ensure label is integer
    if df["label"].dtype != "int64":
        df["label"] = df["label"].astype(int)

    # Fill missing numeric values with median
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "label"]

    print(f"[INFO] Filling missing numeric cols with median: {numeric_cols}")
    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # Fill missing text with empty string
    if "msg_content" in df.columns:
        df["msg_content"] = df["msg_content"].fillna("")

    # Convert num_roles to integer (roles count)
    if "num_roles" in df.columns:
        df["num_roles"] = df["num_roles"].fillna(0).astype(int)

    return df


def basic_eda(df: pd.DataFrame) -> None:
    print("\n[INFO] Basic EDA")
    print("-----------------------------")
    print("[INFO] Head:")
    print(df.head(), "\n")

    print("[INFO] Class balance (label):")
    print(df["label"].value_counts(normalize=True).rename("proportion"))
    print()

    print("[INFO] Summary stats:")
    print(df.describe(include="all"))


def main():
    project_root = Path(__file__).resolve().parents[1]
    raw_path = project_root / "data" / "discord.csv"
    cleaned_path = project_root / "data" / "discord_cleaned.csv"

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. "
            "Copy your file from 'D:\\2 Level\\Case Studies\\Group\\discord.csv' to this location."
        )

    df_raw = load_raw_data(str(raw_path))
    df_clean = clean_data(df_raw)
    basic_eda(df_clean)

    os.makedirs(cleaned_path.parent, exist_ok=True)
    df_clean.to_csv(cleaned_path, index=False)
    print(f"\n[SUCCESS] Cleaned data saved to: {cleaned_path}")
    print(f"[INFO] Cleaned shape: {df_clean.shape}")


if __name__ == "__main__":
    main()
