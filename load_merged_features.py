import os
import json
from urllib.parse import quote_plus
import pandas as pd
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv

load_dotenv()

password = os.getenv("MONGO_PASSWORD")
escaped_password = quote_plus(password)
uri = f"mongodb+srv://aditiuser:{escaped_password}@cluster0.6opbt4j.mongodb.net/weatherwear?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(uri)
db = client["weatherwear"]

MERGED_PATH = "data/processed/merged_dataset.parquet"  # adjust if needed

merged_df = pd.read_parquet(MERGED_PATH)
print(f"Loaded {len(merged_df)} rows from parquet")
print("Columns:", merged_df.columns.tolist())

records = json.loads(merged_df.to_json(orient="records", date_format="iso"))

db.merged_features.drop()
db.merged_features.insert_many(records, ordered=False)

db.merged_features.create_index([("city", ASCENDING)])
if "clothing_category" in merged_df.columns:
    db.merged_features.create_index([("clothing_category", ASCENDING)])

print(f"Inserted {db.merged_features.count_documents({})} merged_features")
print("Sample document:", db.merged_features.find_one())
print("Indexes:", list(db.merged_features.index_information().keys()))

# ── FINAL VERIFICATION ACROSS ALL 5 COLLECTIONS ──
print("\n" + "=" * 50)
print("PHASE 4 VERIFICATION SUMMARY")
print("=" * 50)
for coll_name in ["cities", "weather_readings", "clothing_reviews", "fashion_trends", "merged_features"]:
    count = db[coll_name].count_documents({})
    indexes = list(db[coll_name].index_information().keys())
    print(f"{coll_name}: {count} docs | indexes: {indexes}")

client.close()