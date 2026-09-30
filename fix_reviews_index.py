import os
from urllib.parse import quote_plus
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv

load_dotenv()
password = os.getenv("MONGO_PASSWORD")
escaped_password = quote_plus(password)
uri = f"mongodb+srv://aditiuser:{escaped_password}@cluster0.6opbt4j.mongodb.net/weatherwear?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(uri)
db = client["weatherwear"]

db.clothing_reviews.create_index([("clothing_category", ASCENDING)])
print("Indexes now:", list(db.clothing_reviews.index_information().keys()))
client.close()