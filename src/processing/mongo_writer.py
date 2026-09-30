import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
from pymongo import MongoClient, ASCENDING
from tqdm import tqdm
from config.config import (
    MONGO_URI,
    MONGO_DB_NAME,
    MONGO_COLLECTIONS,
    WEATHER_PROCESSED,
    REVIEWS_PROCESSED,
    FASHION_PROCESSED,
    MERGED_DATASET,
)

def get_db():
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    return client, db

def insert_collection(db, collection_name, df, chunk_size=500):
    col = db[collection_name]
    col.drop()
    records = df.to_dict("records")
    total = 0
    for i in tqdm(range(0, len(records), chunk_size),
                  desc=f"Writing {collection_name}"):
        chunk = records[i:i + chunk_size]
        col.insert_many(chunk)
        total += len(chunk)
    print(f"Inserted {total:,} records into {collection_name}")
    return total

def run_mongo_writer():
    print("Connecting to MongoDB Atlas...")
    client, db = get_db()
    print(f"Connected to: {MONGO_DB_NAME}")

    # Weather records
    print("\nLoading weather_processed.parquet...")
    weather_df = pd.read_parquet(WEATHER_PROCESSED)
    weather_df["date"] = weather_df["date"].astype(str)
    print(f"Weather rows: {len(weather_df):,}")
    insert_collection(db, MONGO_COLLECTIONS["weather"], weather_df)
    col = db[MONGO_COLLECTIONS["weather"]]
    col.create_index([("city", ASCENDING), ("date", ASCENDING)])
    print("Index created on city + date")

    # Reviews
    print("\nLoading reviews_processed.parquet...")
    reviews_df = pd.read_parquet(REVIEWS_PROCESSED)
    print(f"Reviews rows: {len(reviews_df):,}")
    insert_collection(db, MONGO_COLLECTIONS["reviews"], reviews_df)
    col = db[MONGO_COLLECTIONS["reviews"]]
    col.create_index([("year", ASCENDING),
                      ("month", ASCENDING),
                      ("clothing_category", ASCENDING)])
    print("Index created on year + month + clothing_category")

    # Fashion
    print("\nLoading fashion_processed.parquet...")
    fashion_df = pd.read_parquet(FASHION_PROCESSED)
    print(f"Fashion rows: {len(fashion_df):,}")
    insert_collection(db, MONGO_COLLECTIONS["fashion"], fashion_df)

    # Merged
    print("\nLoading merged_dataset.parquet...")
    merged_df = pd.read_parquet(MERGED_DATASET)
    print(f"Merged rows: {len(merged_df):,}")
    insert_collection(db, MONGO_COLLECTIONS["merged"], merged_df)
    col = db[MONGO_COLLECTIONS["merged"]]
    col.create_index([("city", ASCENDING),
                      ("year", ASCENDING),
                      ("month", ASCENDING)])
    print("Index created on city + year + month")

    # Summary
    print("\n" + "=" * 50)
    print("MONGODB WRITE SUMMARY")
    print("=" * 50)
    for name, col_name in MONGO_COLLECTIONS.items():
        if col_name == "recommendations":
            continue
        count = db[col_name].count_documents({})
        print(f"{col_name}: {count:,} documents")

    client.close()
    print("\nMongoDB write complete.")

if __name__ == "__main__":
    run_mongo_writer()