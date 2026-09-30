import os
import json
from urllib.parse import quote_plus
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

password = os.getenv("MONGO_PASSWORD")
escaped_password = quote_plus(password)
uri = f"mongodb+srv://aditiuser:{escaped_password}@cluster0.6opbt4j.mongodb.net/weatherwear?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(uri)
db = client["weatherwear"]

FASHION_PATH = "data/processed/fashion_processed.parquet"  # adjust if needed

fashion_df = pd.read_parquet(FASHION_PATH)
print(f"Loaded {len(fashion_df)} rows from parquet")
print("Columns:", fashion_df.columns.tolist())

records = json.loads(fashion_df.to_json(orient="records", date_format="iso"))

db.fashion_trends.drop()
db.fashion_trends.insert_many(records, ordered=False)

print(f"Inserted {db.fashion_trends.count_documents({})} fashion_trends")
print("Sample document:", db.fashion_trends.find_one())

client.close()
