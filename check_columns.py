import pandas as pd

WEATHER_PATH = "data/processed/weather_processed.parquet"
weather_df = pd.read_parquet(WEATHER_PATH)

print("Columns:", weather_df.columns.tolist())
print("\nFirst 3 rows:")
print(weather_df.head(3))