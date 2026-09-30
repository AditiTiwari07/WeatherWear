import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
from config.config import (
    KAGGLE_DATASET,
    FASHION_RAW_DIR,
)

def run_ingestion():
    print("Starting Fashion Trend data ingestion...")

    FASHION_RAW_DIR.mkdir(parents=True, exist_ok=True)

    os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME", "")
    os.environ["KAGGLE_KEY"]      = os.getenv("KAGGLE_KEY", "")

    import kaggle
    kaggle.api.authenticate()

    print(f"Downloading {KAGGLE_DATASET} from Kaggle...")
    kaggle.api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(FASHION_RAW_DIR),
        unzip=True,
    )

    xls_files = list(FASHION_RAW_DIR.rglob("*.xls")) + list(FASHION_RAW_DIR.rglob("*.xlsx"))
    if not xls_files:
        print("ERROR: No Excel files found.")
        return

    print(f"Using: {xls_files[0].name}")
    df = pd.read_excel(xls_files[0])
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")

    filepath = FASHION_RAW_DIR / "fashion_trends.parquet"
    df.to_parquet(filepath, index=False)

    print(f"Saved -> {filepath}")
    print("Fashion ingestion complete.")

if __name__ == "__main__":
    run_ingestion()