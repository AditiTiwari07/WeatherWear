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

REVIEWS_PATH = "data/processed/reviews_processed.parquet"  # adjust if needed

reviews_df = pd.read_parquet(REVIEWS_PATH)
print(f"Loaded {len(reviews_df)} rows from parquet")
print("Columns:", reviews_df.columns.tolist())

records = json.loads(reviews_df.to_json(orient="records", date_format="iso"))

db.clothing_reviews.drop()
db.clothing_reviews.insert_many(records, ordered=False)

# Index on whichever grouping fields actually exist
possible_index_fields = [f for f in ["category", "year", "month"] if f in reviews_df.columns]
if possible_index_fields:
    db.clothing_reviews.create_index([(f, ASCENDING) for f in possible_index_fields])

print(f"Inserted {db.clothing_reviews.count_documents({})} clothing_reviews")
print("Sample document:", db.clothing_reviews.find_one())
print("Indexes:", list(db.clothing_reviews.index_information().keys()))

client.close()
