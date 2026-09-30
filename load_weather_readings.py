import os
import json
from urllib.parse import quote_plus
import pandas as pd
from pymongo import MongoClient, ASCENDING
from pymongo.errors import BulkWriteError
from dotenv import load_dotenv

load_dotenv()

password = os.getenv("MONGO_PASSWORD")
escaped_password = quote_plus(password)
uri = f"mongodb+srv://aditiuser:{escaped_password}@cluster0.6opbt4j.mongodb.net/weatherwear?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(uri)
db = client["weatherwear"]

WEATHER_PATH = "data/processed/weather_processed.parquet"  # same path as Step 4
BATCH_SIZE = 5000

weather_df = pd.read_parquet(WEATHER_PATH)
print(f"Loaded {len(weather_df)} rows from parquet")

records = json.loads(weather_df.to_json(orient="records", date_format="iso"))

db.weather_readings.drop()  # fresh load

total = len(records)
inserted = 0
for i in range(0, total, BATCH_SIZE):
    batch = records[i:i + BATCH_SIZE]
    try:
        db.weather_readings.insert_many(batch, ordered=False)
        inserted += len(batch)
    except BulkWriteError as e:
        failed = len(e.details.get("writeErrors", []))
        inserted += len(batch) - failed
        print(f"  Warning: {failed} docs failed in batch starting at {i}")
    print(f"  Progress: {inserted}/{total}", end="\r")

print(f"\nInserted {db.weather_readings.count_documents({})} weather readings")

# city + date compound index, matching what the timeline asked for
db.weather_readings.create_index([("city", ASCENDING), ("date", ASCENDING)], name="city_date_idx")

print("Sample document:", db.weather_readings.find_one())
print("Indexes:", list(db.weather_readings.index_information().keys()))

client.close()