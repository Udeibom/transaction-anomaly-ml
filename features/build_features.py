# features/build_features.py

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def build_features(
    input_csv: Path,
    output_path: Path,
    scaler_path: Path,
    rolling_window: int = 10
):
    """
    Build behavioral features for transaction anomaly detection.

    Parameters
    ----------
    input_csv : str
        Path to raw transactions CSV
    output_path : str
        Path to save processed feature dataframe (parquet or csv)
    scaler_path : str
        Path to save fitted scaler
    rolling_window : int
        Number of past transactions per card for rolling stats
    """

    # -----------------------------
    # Load & sort
    # -----------------------------
    df = pd.read_csv(
        input_csv,
        parse_dates=["trans_date_trans_time"],
        low_memory=False
    )

    df = df.sort_values(["cc_num", "trans_date_trans_time"])

    # -----------------------------
    # Log-scaled amount
    # -----------------------------
    df["log_amt"] = np.log1p(df["amt"])

    # -----------------------------
    # Time-based features
    # -----------------------------
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek

    # Time since last transaction (per card)
    df["time_since_last_txn"] = (
        df.groupby("cc_num")["trans_date_trans_time"]
        .diff()
        .dt.total_seconds()
    )

    # Fill first transaction gaps
    df["time_since_last_txn"] = df["time_since_last_txn"].fillna(0)

    # -----------------------------
    # Rolling behavioral stats (per card)
    # IMPORTANT: shift to avoid leakage
    # -----------------------------
    grouped = df.groupby("cc_num")

    df["rolling_mean_amt"] = (
        grouped["amt"]
        .shift(1)
        .rolling(rolling_window)
        .mean()
    )

    df["rolling_std_amt"] = (
        grouped["amt"]
        .shift(1)
        .rolling(rolling_window)
        .std()
    )

    df["rolling_txn_count"] = (
        grouped["amt"]
        .shift(1)
        .rolling(rolling_window)
        .count()
    )

    # Fill NaNs from early transactions
    df[[
        "rolling_mean_amt",
        "rolling_std_amt",
        "rolling_txn_count"
    ]] = df[[
        "rolling_mean_amt",
        "rolling_std_amt",
        "rolling_txn_count"
    ]].fillna(0)

    # -----------------------------
    # Amount deviation (z-score)
    # -----------------------------
    df["amt_zscore_card"] = (
        (df["amt"] - df["rolling_mean_amt"]) /
        (df["rolling_std_amt"] + 1e-6)
    )

    # -----------------------------
    # Select final feature set
    # -----------------------------
    feature_cols = [
        "log_amt",
        "time_since_last_txn",
        "hour",
        "day_of_week",
        "rolling_mean_amt",
        "rolling_std_amt",
        "rolling_txn_count",
        "amt_zscore_card",
        "city_pop",
        "lat", "long",
        "merch_lat", "merch_long"
    ]

    X = df[feature_cols].replace([np.inf, -np.inf], 0)

    # -----------------------------
    # Normalize
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled_df = pd.DataFrame(
        X_scaled,
        columns=feature_cols,
        index=df.index
    )

    # -----------------------------
    # Save artifacts
    # -----------------------------
    X_scaled_df.to_parquet(output_path, index=False)
    joblib.dump(scaler, scaler_path)

    print("✅ Feature engineering complete")
    print(f"Saved features → {output_path}")
    print(f"Saved scaler → {scaler_path}")


if __name__ == "__main__":
    build_features(
        input_csv=DATA_DIR / "creditcard.csv",
        output_path=PROCESSED_DIR / "features.parquet",
        scaler_path=PROCESSED_DIR / "feature_scaler.joblib"
    )

