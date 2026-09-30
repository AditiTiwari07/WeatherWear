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


WEATHER_PATH = "data/processed/weather_processed.parquet"

weather_df = pd.read_parquet(WEATHER_PATH)


if "latitude" in weather_df.columns and "longitude" in weather_df.columns:
    cities_df = weather_df[["city", "latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
else:
    cities_df = weather_df[["city"]].drop_duplicates().reset_index(drop=True)

records = json.loads(cities_df.to_json(orient="records"))

db.cities.drop()  
db.cities.insert_many(records)
db.cities.create_index([("city", ASCENDING)], unique=True)

print(f"Inserted {db.cities.count_documents({})} cities")
print("Sample document:", db.cities.find_one())

client.close()