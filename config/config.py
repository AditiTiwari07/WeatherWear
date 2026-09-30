import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR       = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"
OUTPUTS_DIR   = BASE_DIR / "outputs"

WEATHER_RAW_DIR = RAW_DIR / "weather"
REVIEWS_RAW_DIR = RAW_DIR / "reviews"
FASHION_RAW_DIR = RAW_DIR / "fashion"
WEATHER_PROCESSED   = PROCESSED_DIR / "weather_processed.parquet"
REVIEWS_PROCESSED   = PROCESSED_DIR / "reviews_processed.parquet"
FASHION_PROCESSED   = PROCESSED_DIR / "fashion_processed.parquet"
MERGED_DATASET      = PROCESSED_DIR / "merged_dataset.parquet"
CITIES = [
    {"name": "Dublin",      "lat": 53.3498,  "lon": -6.2603},
    {"name": "London",      "lat": 51.5074,  "lon": -0.1278},
    {"name": "New York",    "lat": 40.7128,  "lon": -74.0060},
    {"name": "Tokyo",       "lat": 35.6762,  "lon": 139.6503},
    {"name": "Sydney",      "lat": -33.8688, "lon": 151.2093},
    {"name": "Mumbai",      "lat": 19.0760,  "lon": 72.8777},
    {"name": "Paris",       "lat": 48.8566,  "lon": 2.3522},
    {"name": "Berlin",      "lat": 52.5200,  "lon": 13.4050},
    {"name": "Toronto",     "lat": 43.6532,  "lon": -79.3832},
    {"name": "Dubai",       "lat": 25.2048,  "lon": 55.2708},
    {"name": "Sao Paulo",   "lat": -23.5505, "lon": -46.6333},
    {"name": "Singapore",   "lat": 1.3521,   "lon": 103.8198},
    {"name": "Cape Town",   "lat": -33.9249, "lon": 18.4241},
    {"name": "Moscow",      "lat": 55.7558,  "lon": 37.6173},
    {"name": "Chicago",     "lat": 41.8781,  "lon": -87.6298},
    {"name": "Seoul",       "lat": 37.5665,  "lon": 126.9780},
    {"name": "Mexico City", "lat": 19.4326,  "lon": -99.1332},
    {"name": "Jakarta",     "lat": -6.2088,  "lon": 106.8456},
    {"name": "Cairo",       "lat": 30.0444,  "lon": 31.2357},
    {"name": "Stockholm",   "lat": 59.3293,  "lon": 18.0686},
]

WEATHER_VARIABLES = [
    "temperature_2m",
    "precipitation",
    "windspeed_10m",
    "relativehumidity_2m",
    "uv_index",
    "weathercode",
]

WEATHER_START_DATE = "2019-01-01"
WEATHER_END_DATE   = "2023-12-31"

HF_DATASET_NAME   = "McAuley-Lab/Amazon-Reviews-2023"
HF_DATASET_SUBSET = "raw_review_Clothing_Shoes_and_Jewelry"
REVIEWS_SAMPLE_SIZE = 500000

KAGGLE_DATASET  = "fashionworldda/fashion-trend-dataset"
FASHION_CSV_NAME = "fashion_data.csv"

MONGO_URI     = os.getenv("MONGO_URI")
MONGO_DB_NAME = "weatherwear"

MONGO_COLLECTIONS = {
    "weather":     "weather_records",
    "reviews":     "clothing_reviews",
    "fashion":     "fashion_trends",
    "merged":      "merged_data",
    "predictions": "recommendations",
}

CLOTHING_CATEGORIES = [
    "Light clothing",
    "Casual wear",
    "Layered clothing",
    "Heavy outerwear",
    "Waterproof clothing",
    "UV-protective wear",
    "Thermal wear",
]

WEATHER_FEATURES = [
    "temperature_2m",
    "precipitation",
    "windspeed_10m",
    "relativehumidity_2m",
    "uv_index",
    "season",
    "month",
]

MODEL_CLASSIFIER_PATH = MODELS_DIR / "clothing_classifier.pkl"
MODEL_FORECASTER_PATH = MODELS_DIR / "demand_forecaster.pkl"
LABEL_ENCODER_PATH    = MODELS_DIR / "label_encoder.pkl"

TEST_SIZE    = 0.2
RANDOM_STATE = 42

DASH_HOST  = "127.0.0.1"
DASH_PORT  = 8050
DASH_DEBUG = True

def create_dirs():
    for d in [WEATHER_RAW_DIR, REVIEWS_RAW_DIR, FASHION_RAW_DIR,
              PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print("All directories verified.")

if __name__ == "__main__":
    create_dirs()
    print(f"Project root: {BASE_DIR}")
    print(f"Cities configured: {len(CITIES)}")