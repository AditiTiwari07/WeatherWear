import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from tqdm import tqdm
from config.config import (
    CITIES,
    WEATHER_VARIABLES,
    WEATHER_START_DATE,
    WEATHER_END_DATE,
    WEATHER_RAW_DIR,
)

def setup_client():
    cache_session = requests_cache.CachedSession(
        '.cache', expire_after=-1
    )
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)
    return client

def fetch_city_weather(client, city):
    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "start_date": WEATHER_START_DATE,
        "end_date": WEATHER_END_DATE,
        "hourly": WEATHER_VARIABLES,
    }
    responses = client.weather_api(
        "https://archive-api.open-meteo.com/v1/archive",
        params=params
    )
    response = responses[0]
    hourly = response.Hourly()

    data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m":       hourly.Variables(0).ValuesAsNumpy(),
        "precipitation":        hourly.Variables(1).ValuesAsNumpy(),
        "windspeed_10m":        hourly.Variables(2).ValuesAsNumpy(),
        "relativehumidity_2m":  hourly.Variables(3).ValuesAsNumpy(),
        "uv_index":             hourly.Variables(4).ValuesAsNumpy(),
        "weathercode":          hourly.Variables(5).ValuesAsNumpy(),
    }

    df = pd.DataFrame(data)
    df["city"] = city["name"]
    df["latitude"] = city["lat"]
    df["longitude"] = city["lon"]
    return df

def run_ingestion():
    print("Starting weather ingestion...")
    client = setup_client()
    WEATHER_RAW_DIR.mkdir(parents=True, exist_ok=True)

    for city in tqdm(CITIES, desc="Fetching cities"):
        try:
            df = fetch_city_weather(client, city)
            filename = city["name"].replace(" ", "_").lower() + ".csv"
            filepath = WEATHER_RAW_DIR / filename
            df.to_csv(filepath, index=False)
            print(f"Saved {city['name']}: {len(df)} rows -> {filepath}")
            time.sleep(7)
        except Exception as e:
            print(f"ERROR fetching {city['name']}: {e}")
            time.sleep(65)

    print("Weather ingestion complete.")

if __name__ == "__main__":
    run_ingestion()