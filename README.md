# WeatherWear

Big Data clothing recommendation and demand forecasting system.

MSc Big Data Management and Analytics, Griffith College Dublin

Student: Aditi Tiwari (Student No: 3195197)


## Overview

WeatherWear recommends clothing to consumers based on live and historical
weather, and forecasts clothing demand for manufacturers by city, month,
and category. It combines weather data, e-commerce clothing reviews, and
fashion trend data, processed with PySpark, stored in MongoDB Atlas,
modelled with scikit-learn and XGBoost, and served through a Plotly Dash
dashboard.

## Tech Stack

- Python 3.11
- PySpark 4.0.3 (Google Colab)
- MongoDB Atlas (pymongo)
- scikit-learn (Random Forest classifier)
- XGBoost (demand forecaster)
- Plotly Dash + Dash Bootstrap Components
- Open-Meteo API

## Project Structure

```
WeatherWear/
├── config/
│   └── config.py            # paths, MongoDB URI, model paths, Dash settings
├── data/
│   ├── raw/                 # raw ingested data
│   └── processed/           # PySpark outputs (parquet)
├── models/
│   ├── clothing_classifier.pkl
│   ├── label_encoder.pkl
│   ├── demand_forecaster.pkl
│   └── city_encoder.pkl
├── src/
│   └── dashboard/
│       └── app.py           # Plotly Dash application
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## Data Sources

| Source | Method | Volume |
|---|---|---|
| Open-Meteo Weather API | REST API, 20 cities | 876,480 rows |
| Amazon Reviews 2023 (clothing) | HuggingFace streaming | 326,493 rows |
| Kaggle Fashion Trends | Static dataset | 660 rows |

## Pipeline Summary

- Ingested weather, review, and fashion data into `data/raw/`.
- Processed with PySpark on Google Colab: added season and clothing-label
  features, aggregated reviews to monthly demand, joined all sources into
  `merged_dataset.parquet` (1,200 rows, 15 columns).
- EDA: dropped `avg_uv_index`, `demand_score_norm`, and `price` (100%
  null). Found r=0.36 between temperature and demand, and class
  imbalance in clothing category.
- Loaded processed collections into MongoDB Atlas. Raw hourly weather
  (876K rows) kept as local parquet due to the free-tier 512MB limit.
- Trained a Random Forest classifier (97.06% accuracy, F1 0.97) for
  clothing recommendation, and an XGBoost regressor (R² 0.37) for demand
  forecasting.
- Built a Plotly Dash dashboard serving both models: consumer outfit
  recommendation and manufacturer demand forecast.

## Running the Dashboard

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/dashboard/app.py
```

Then open `http://127.0.0.1:8050` in a browser. Requires a valid
`MONGO_URI` in `.env`.

## Dashboard Design Notes

- Live weather is always shown in a "Right now in [city]" strip at the
  top, independent of the month selector.
- The KPI cards, outfit recommendation, and demand forecast always
  reflect the historical average for the selected month, including the
  current month, so their meaning never changes based on the dropdown.
- The 7-day forecast chart is explicitly live.

## Known Limitations

- The XGBoost model predicts a single aggregate demand figure per
  city/month; the split across clothing categories uses a temperature-
  based weighting formula rather than a per-category model prediction,
  due to the lack of per-category demand labels in training data.
- `avg_uv_index`, `demand_score_norm`, and `price` were dropped from
  model training due to being 100% null.
