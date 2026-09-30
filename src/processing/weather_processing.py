import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType, TimestampType
from config.config import (
    WEATHER_RAW_DIR,
    WEATHER_PROCESSED,
    PROCESSED_DIR,
)

def create_spark():
    spark = SparkSession.builder \
        .appName("WeatherWear-WeatherProcessing") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def add_season(df):
    return df.withColumn("season",
        F.when(F.col("month").isin(12, 1, 2), 3)
         .when(F.col("month").isin(3, 4, 5), 0)
         .when(F.col("month").isin(6, 7, 8), 1)
         .otherwise(2)
    )

def add_clothing_label(df):
    return df.withColumn("clothing_label",
        F.when((F.col("temperature_2m") > 25) & (F.col("precipitation") < 1), "Light clothing")
         .when((F.col("temperature_2m") > 15) & (F.col("precipitation") >= 1), "Waterproof clothing")
         .when((F.col("temperature_2m") > 15), "Casual wear")
         .when((F.col("temperature_2m") > 5) & (F.col("precipitation") >= 1), "Waterproof clothing")
         .when((F.col("temperature_2m") > 5), "Layered clothing")
         .when((F.col("temperature_2m") > -5), "Heavy outerwear")
         .otherwise("Thermal wear")
    )

def run_processing():
    print("Starting weather processing...")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    spark = create_spark()

    csv_files = list(WEATHER_RAW_DIR.glob("*.csv"))
    print(f"Loading {len(csv_files)} city files...")

    df = spark.read.csv(
        [str(f) for f in csv_files],
        header=True,
        inferSchema=True,
    )

    print(f"Raw rows: {df.count():,}")

    df = df.withColumn("date", F.col("date").cast(TimestampType()))
    df = df.withColumn("temperature_2m", F.col("temperature_2m").cast(FloatType()))
    df = df.withColumn("precipitation", F.col("precipitation").cast(FloatType()))
    df = df.withColumn("windspeed_10m", F.col("windspeed_10m").cast(FloatType()))
    df = df.withColumn("relativehumidity_2m", F.col("relativehumidity_2m").cast(FloatType()))
    df = df.withColumn("uv_index", F.col("uv_index").cast(FloatType()))
    df = df.withColumn("weathercode", F.col("weathercode").cast(IntegerType()))

    df = df.withColumn("year", F.year("date"))
    df = df.withColumn("month", F.month("date"))

    df = add_season(df)
    df = add_clothing_label(df)

    df = df.dropna(subset=[
        "temperature_2m", "precipitation",
        "windspeed_10m", "relativehumidity_2m"
    ])

    print(f"Processed rows: {df.count():,}")
    print("Sample:")
    df.select("city", "date", "temperature_2m", "season", "clothing_label").show(5)

    df.write.mode("overwrite").parquet(str(WEATHER_PROCESSED))
    print(f"Saved -> {WEATHER_PROCESSED}")
    print("Weather processing complete.")

    spark.stop()

if __name__ == "__main__":
    run_processing()