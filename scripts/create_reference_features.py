import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURES_PATH = BASE_DIR / "data/processed/features.parquet"
REF_PATH = BASE_DIR / "data/processed/reference_features.parquet"

df = pd.read_parquet(FEATURES_PATH)

# Use first 20% as reference window
ref_df = df.iloc[: int(0.2 * len(df))]

REF_PATH.parent.mkdir(parents=True, exist_ok=True)
ref_df.to_parquet(REF_PATH)

print(f"✅ Reference features saved to {REF_PATH}")
