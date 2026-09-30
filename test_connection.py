import os
from urllib.parse import quote_plus
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

password = os.getenv("MONGO_PASSWORD")
escaped_password = quote_plus(password)
uri = f"mongodb+srv://aditiuser:{escaped_password}@cluster0.6opbt4j.mongodb.net/weatherwear?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(uri)
db = client["weatherwear"]

print("Connected successfully!")
print("Existing collections:", db.list_collection_names())

client.close()