import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from config.config import (
    REVIEWS_SAMPLE_SIZE,
    REVIEWS_RAW_DIR,
)

def run_ingestion():
    print("Starting Amazon Reviews ingestion...")
    print(f"Streaming {REVIEWS_SAMPLE_SIZE:,} rows from HuggingFace...")

    REVIEWS_RAW_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_Clothing_Shoes_and_Jewelry",
        split="full",
        streaming=True,
        trust_remote_code=True,
    )

    rows = []
    for i, record in enumerate(tqdm(dataset, desc="Streaming reviews", total=REVIEWS_SAMPLE_SIZE)):
        if i >= REVIEWS_SAMPLE_SIZE:
            break
        rows.append({
            "asin":              record.get("asin", ""),
            "title":             record.get("title", ""),
            "rating":            record.get("rating", None),
            "timestamp":         record.get("timestamp", None),
            "verified_purchase": record.get("verified_purchase", None),
            "helpful_vote":      record.get("helpful_vote", None),
            "text":              record.get("text", ""),
        })

    print("Converting to DataFrame...")
    df = pd.DataFrame(rows)

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    df["year"]  = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    df = df.dropna(subset=["rating", "timestamp"])
    df = df[df["year"].between(2019, 2023)]

    filepath = REVIEWS_RAW_DIR / "clothing_reviews.parquet"
    df.to_parquet(filepath, index=False)

    print(f"Saved {len(df):,} reviews -> {filepath}")
    print("Reviews ingestion complete.")

if __name__ == "__main__":
    run_ingestion()