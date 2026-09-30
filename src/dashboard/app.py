import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import requests
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
from pymongo import MongoClient
from dotenv import load_dotenv
import datetime

load_dotenv()

from config.config import (
    MONGO_URI, MONGO_DB_NAME, MONGO_COLLECTIONS,
    MODEL_CLASSIFIER_PATH, LABEL_ENCODER_PATH,
    MODEL_FORECASTER_PATH,
    MODELS_DIR, DASH_HOST, DASH_PORT, DASH_DEBUG,
)

clf = joblib.load(MODEL_CLASSIFIER_PATH)
le = joblib.load(LABEL_ENCODER_PATH)
le_city = joblib.load(MODELS_DIR / "city_encoder.pkl")
forecaster = joblib.load(MODEL_FORECASTER_PATH)

client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]

CITIES = {
    "Dublin":      (53.3498, -6.2603),
    "London":      (51.5074, -0.1278),
    "New York":    (40.7128, -74.0060),
    "Tokyo":       (35.6762, 139.6503),
    "Sydney":      (-33.8688, 151.2093),
    "Mumbai":      (19.0760, 72.8777),
    "Paris":       (48.8566, 2.3522),
    "Berlin":      (52.5200, 13.4050),
    "Toronto":     (43.6532, -79.3832),
    "Dubai":       (25.2048, 55.2708),
    "Sao Paulo":   (-23.5505, -46.6333),
    "Singapore":   (1.3521, 103.8198),
    "Cape Town":   (-33.9249, 18.4241),
    "Moscow":      (55.7558, 37.6173),
    "Chicago":     (41.8781, -87.6298),
    "Seoul":       (37.5665, 126.9780),
    "Mexico City": (19.4326, -99.1332),
    "Jakarta":     (-6.2088, 106.8456),
    "Cairo":       (30.0444, 31.2357),
    "Stockholm":   (59.3293, 18.0686),
}

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March",
    4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September",
    10: "October", 11: "November", 12: "December"
}

WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy",
    3: "Overcast", 45: "Foggy", 48: "Icy fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    61: "Slight rain", 63: "Rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Snow", 75: "Heavy snow",
    80: "Slight showers", 81: "Showers", 82: "Heavy showers",
    95: "Thunderstorm", 99: "Thunderstorm with hail",
}

CLOTHING_DETAILS = {
    "Light clothing": {
        "emoji": "☀️",
        "suggestion": "Wear a light t-shirt, shorts, or a summer dress.",
        "items": "T-shirt, shorts, sundress, sandals",
        "color": "#FF8C00",
    },
    "Casual wear": {
        "emoji": "👕",
        "suggestion": "A casual outfit works perfectly today.",
        "items": "Jeans, polo shirt, light hoodie, sneakers",
        "color": "#1D9E75",
    },
    "Layered clothing": {
        "emoji": "🧥",
        "suggestion": "Layer up today. Start light and add a jacket.",
        "items": "Long sleeve top, cardigan or light jacket, jeans",
        "color": "#5B8DB8",
    },
    "Heavy outerwear": {
        "emoji": "🧣",
        "suggestion": "It is cold outside. Wear a warm coat and scarf.",
        "items": "Winter coat, scarf, gloves, warm trousers, boots",
        "color": "#7B5EA7",
    },
    "Waterproof clothing": {
        "emoji": "🌧️",
        "suggestion": "Rain expected. Grab a waterproof jacket or umbrella.",
        "items": "Raincoat, waterproof boots, umbrella",
        "color": "#2E86AB",
    },
    "Thermal wear": {
        "emoji": "❄️",
        "suggestion": "Very cold. Wear thermal layers and a heavy coat.",
        "items": "Thermal base layer, heavy coat, fur-lined boots, hat, gloves",
        "color": "#4A90D9",
    },
    "UV-protective wear": {
        "emoji": "🕶️",
        "suggestion": "High UV today. Protect your skin.",
        "items": "UV-protective shirt, wide brim hat, sunglasses",
        "color": "#E8A838",
    },
}

CATEGORIES = [
    "Light clothing",
    "Casual wear",
    "Layered clothing",
    "Heavy outerwear",
    "Waterproof clothing",
    "Thermal wear",
]

def get_live_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": [
            "temperature_2m",
            "precipitation",
            "windspeed_10m",
            "relativehumidity_2m",
            "weathercode",
        ],
        "hourly": "temperature_2m",
        "forecast_days": 7,
    }
    r = requests.get(url, params=params, timeout=10)
    data = r.json()
    return data["current"], data["hourly"]

def predict_clothing(temp, precip, wind, humidity, month, season,
                     avg_rating=4.46, review_count=1000,
                     review_demand_score=0.19):
    features = pd.DataFrame([[
        temp, precip, wind, humidity,
        month, season,
        avg_rating, review_count, review_demand_score
    ]], columns=[
        "avg_temperature", "avg_precipitation",
        "avg_windspeed", "avg_humidity",
        "month", "season",
        "avg_rating", "review_count",
        "review_demand_score"
    ])
    pred = clf.predict(features)[0]
    proba = clf.predict_proba(features)[0]
    confidence = round(float(proba.max()) * 100, 1)
    label = le.inverse_transform([pred])[0]
    return label, confidence

def predict_demand(temp, precip, wind, humidity, month, season, city):
    city_encoded = le_city.transform([city])[0]
    features = pd.DataFrame([[
        temp, precip, wind, humidity,
        month, season, city_encoded
    ]], columns=[
        "avg_temperature", "avg_precipitation",
        "avg_windspeed", "avg_humidity",
        "month", "season", "city_encoded"
    ])
    demand = forecaster.predict(features)[0]
    return max(0, round(float(demand)))

def get_season(month):
    if month in [12, 1, 2]:
        return 3
    if month in [3, 4, 5]:
        return 0
    if month in [6, 7, 8]:
        return 1
    return 2

def weather_description(temp, precip, wind, weathercode=None):
    condition = WEATHER_CODES.get(int(weathercode), "") if weathercode is not None else ""
    if temp >= 28:
        temp_desc = "Hot"
    elif temp >= 20:
        temp_desc = "Warm"
    elif temp >= 12:
        temp_desc = "Mild"
    elif temp >= 5:
        temp_desc = "Cool"
    else:
        temp_desc = "Cold"
    if wind >= 30:
        wind_desc = "very windy"
    elif wind >= 15:
        wind_desc = "breezy"
    else:
        wind_desc = ""
    parts = [temp_desc]
    if condition:
        parts.append(condition.lower())
    if wind_desc:
        parts.append(wind_desc)
    return " and ".join(parts)

def get_demand_scores(base_demand, temp, precip, recommended_cat):
    temp_modifiers = {
        "Light clothing":
            max(0.05, (temp - 15) / 20),
        "Casual wear":
            max(0.05, 1 - abs(temp - 15) / 15),
        "Layered clothing":
            max(0.05, 1 - abs(temp - 10) / 10),
        "Heavy outerwear":
            max(0.05, (15 - temp) / 15),
        "Waterproof clothing":
            min(1.0, precip * 2 + 0.3),
        "Thermal wear":
            max(0.05, (5 - temp) / 10),
    }
    scores = []
    for cat in CATEGORIES:
        mod = temp_modifiers.get(cat, 0.1)
        if cat == recommended_cat:
            mod = max(mod, 0.9)
        scores.append(round(base_demand * max(0.05, mod)))
    return scores

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "WeatherWear"

CARD_STYLE = {
    "borderRadius": "12px",
    "border": "1px solid #eee",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
}

app.layout = dbc.Container([

    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("WeatherWear",
                        style={"fontWeight": "bold",
                               "color": "#1D9E75",
                               "fontSize": "42px",
                               "marginBottom": "4px"}),
                html.P("Big Data clothing recommendation and demand forecasting system",
                       style={"color": "#666", "fontSize": "15px",
                              "marginBottom": "2px"}),
                html.Div(id="live-clock",
                         style={"color": "#aaa", "fontSize": "13px"}),
                dcc.Interval(
                    id="clock-interval",
                    interval=6000,
                    n_intervals=0,
                ),
            ], className="text-center mt-4 mb-4"),
        ])
    ]),

    dbc.Row([
        dbc.Col([
            html.Div(id="live-strip", className="text-center mb-4"),
        ])
    ]),

    dbc.Row([
        dbc.Col([
            html.Label("Select a City",
                       style={"fontWeight": "600",
                              "marginBottom": "6px",
                              "display": "block"}),
            dcc.Dropdown(
                id="city-dropdown",
                options=[{"label": c, "value": c} for c in CITIES],
                value="Dublin",
                clearable=False,
                style={"fontSize": "15px", "borderRadius": "8px"},
            ),
        ], width=5),
        dbc.Col([
            html.Label("Select a Month",
                       style={"fontWeight": "600",
                              "marginBottom": "6px",
                              "display": "block"}),
            dcc.Dropdown(
                id="month-dropdown",
                options=[
                    {"label": "January",   "value": 1},
                    {"label": "February",  "value": 2},
                    {"label": "March",     "value": 3},
                    {"label": "April",     "value": 4},
                    {"label": "May",       "value": 5},
                    {"label": "June",      "value": 6},
                    {"label": "July",      "value": 7},
                    {"label": "August",    "value": 8},
                    {"label": "September", "value": 9},
                    {"label": "October",   "value": 10},
                    {"label": "November",  "value": 11},
                    {"label": "December",  "value": 12},
                ],
                value=datetime.datetime.now().month,
                clearable=False,
                style={"fontSize": "15px", "borderRadius": "8px"},
            ),
        ], width=4),
        dbc.Col([
            html.Label(".",
                       style={"color": "white",
                              "display": "block",
                              "marginBottom": "6px"}),
            dbc.Button(
                "Get Recommendation",
                id="refresh-btn",
                color="success",
                size="lg",
                style={"width": "100%", "borderRadius": "8px"},
            ),
        ], width=3),
    ], className="mb-4"),

    dbc.Row(id="weather-cards", className="mb-2"),

    html.Div(id="weather-description",
             className="text-center mb-3",
             style={"fontSize": "15px", "color": "#555",
                    "fontStyle": "italic"}),

    dbc.Row([
        dbc.Col([
            html.H5("Consumer Recommendation",
                    style={"fontWeight": "600",
                           "color": "#333",
                           "borderLeft": "4px solid #1D9E75",
                           "paddingLeft": "10px"}),
            html.P("What to wear based on selected city and month",
                   style={"color": "#888", "fontSize": "13px",
                          "marginLeft": "14px"}),
        ])
    ], className="mb-2"),

    dbc.Row([
        dbc.Col([
            html.Div(id="recommendation-card")
        ], width=12),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("7-Day Temperature Forecast (Live)",
                            style={"fontWeight": "600", "color": "#333"}),
                    dcc.Graph(id="forecast-chart",
                              config={"displayModeBar": False}),
                ])
            ], style=CARD_STYLE),
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Historical Clothing Demand by Season",
                            style={"fontWeight": "600", "color": "#333"}),
                    dcc.Graph(id="demand-chart",
                              config={"displayModeBar": False}),
                ])
            ], style=CARD_STYLE),
        ], width=6),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            html.H5("Manufacturer Demand Forecast",
                    style={"fontWeight": "600",
                           "color": "#333",
                           "borderLeft": "4px solid #FF8C00",
                           "paddingLeft": "10px"}),
            html.P("XGBoost predicted clothing demand for selected city and month",
                   style={"color": "#888", "fontSize": "13px",
                          "marginLeft": "14px"}),
        ])
    ], className="mb-2"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Predicted Sales Demand by Clothing Category",
                            style={"fontWeight": "600", "color": "#333"}),
                    html.P("GREEN = HIGH demand  |  ORANGE = MEDIUM  |  GREY = LOW",
                           style={"fontSize": "12px", "color": "#888"}),
                    dcc.Graph(id="demand-forecast-chart",
                              config={"displayModeBar": False}),
                ])
            ], style=CARD_STYLE),
        ], width=12),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Historical Temperature vs Clothing Category",
                            style={"fontWeight": "600", "color": "#333"}),
                    html.P("5 years of historical data from 2019 to 2023",
                           style={"fontSize": "12px", "color": "#888"}),
                    dcc.Graph(id="historical-chart",
                              config={"displayModeBar": False}),
                ])
            ], style=CARD_STYLE),
        ], width=12),
    ], className="mb-5"),

], fluid=True, style={"maxWidth": "1200px",
                      "margin": "0 auto",
                      "fontFamily": "Segoe UI, sans-serif",
                      "backgroundColor": "#FAFAFA",
                      "padding": "0 20px"})


@app.callback(
    Output("live-clock", "children"),
    Output("live-strip", "children"),
    Output("weather-cards", "children"),
    Output("weather-description", "children"),
    Output("recommendation-card", "children"),
    Output("forecast-chart", "figure"),
    Output("demand-chart", "figure"),
    Output("demand-forecast-chart", "figure"),
    Output("historical-chart", "figure"),
    Input("clock-interval", "n_intervals"),
    Input("city-dropdown", "value"),
    Input("month-dropdown", "value"),
    Input("refresh-btn", "n_clicks"),
)
def update_dashboard(n_intervals, city, selected_month, n_clicks):
    lat, lon = CITIES[city]
    current_month = datetime.datetime.now().month
    month = selected_month if selected_month else current_month
    season = get_season(month)

    # --- Live conditions: always real-time, independent of the month
    # dropdown. Powers the "Right now" strip and the 7-day forecast only.
    try:
        current, hourly = get_live_weather(lat, lon)
        live_temp        = current["temperature_2m"]
        live_precip      = current["precipitation"]
        live_wind        = current["windspeed_10m"]
        live_humidity    = current["relativehumidity_2m"]
        live_weathercode = current.get("weathercode", 0)
        live_ok = True
    except Exception:
        live_temp, live_precip, live_wind, live_humidity = 15.0, 0.0, 10.0, 70.0
        live_weathercode = 0
        hourly = {"time": [], "temperature_2m": []}
        live_ok = False

    # --- Monthly average: always pulled from MongoDB for whichever month
    # is selected (including the current month), so the KPI cards below
    # never silently swap between a live reading and a 5-year average.
    temp, precip, wind, humidity = 15.0, 0.0, 10.0, 70.0
    weathercode = None
    try:
        hist_docs = list(db[MONGO_COLLECTIONS["merged"]].find(
            {"city": city, "month": month},
            {"avg_temperature": 1, "avg_precipitation": 1,
             "avg_windspeed": 1, "avg_humidity": 1, "_id": 0}
        ))
        if hist_docs:
            hist_month_df = pd.DataFrame(hist_docs)
            temp     = round(float(
                hist_month_df["avg_temperature"].mean()), 1)
            precip   = round(float(
                hist_month_df["avg_precipitation"].mean()), 2)
            wind     = round(float(
                hist_month_df["avg_windspeed"].mean()), 1)
            humidity = round(float(
                hist_month_df["avg_humidity"].mean()), 0)
        elif live_ok and month == current_month:
            # Fallback only if no historical docs exist yet for this
            # month/city: use live values rather than the hardcoded default.
            temp, precip, wind, humidity = (
                live_temp, live_precip, live_wind, live_humidity)
    except Exception:
        if live_ok:
            temp, precip, wind, humidity = (
                live_temp, live_precip, live_wind, live_humidity)

    now = datetime.datetime.now()
    clock_display = [
        html.Span(f"Today: {now.strftime('%A %d %B %Y')}",
                  style={"marginRight": "16px"}),
        html.Span(f"Last weather fetch: {now.strftime('%H:%M:%S')}",
                  style={"color": "#1D9E75"}),
    ]

    live_desc = weather_description(
        live_temp, live_precip, live_wind, live_weathercode)
    live_strip = html.Div([
        html.Span("LIVE",
                  style={"backgroundColor": "#1D9E75", "color": "white",
                         "fontSize": "10px", "fontWeight": "700",
                         "letterSpacing": "0.08em", "padding": "3px 8px",
                         "borderRadius": "10px", "marginRight": "8px",
                         "verticalAlign": "middle"}),
        html.Span(
            f"Right now in {city}: {live_temp:.1f}C, {live_desc}"
            if live_ok else f"Live weather for {city} unavailable",
            style={"fontSize": "13px", "color": "#555",
                   "verticalAlign": "middle"}),
    ])

    card_style = {
        "borderRadius": "12px",
        "border": "1px solid #eee",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
        "textAlign": "center",
        "padding": "16px",
    }

    weather_cards = [
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H2(f"{temp:.1f}C",
                    style={"fontWeight": "700", "color": "#333",
                           "marginBottom": "2px"}),
            html.P("Avg Temperature",
                   style={"color": "#888", "margin": "0",
                          "fontSize": "13px"}),
        ]), style=card_style), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H2(f"{precip:.1f}mm",
                    style={"fontWeight": "700", "color": "#333",
                           "marginBottom": "2px"}),
            html.P("Avg Precipitation",
                   style={"color": "#888", "margin": "0",
                          "fontSize": "13px"}),
        ]), style=card_style), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H2(f"{wind:.1f}km/h",
                    style={"fontWeight": "700", "color": "#333",
                           "marginBottom": "2px"}),
            html.P("Avg Wind Speed",
                   style={"color": "#888", "margin": "0",
                          "fontSize": "13px"}),
        ]), style=card_style), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H2(f"{humidity:.0f}%",
                    style={"fontWeight": "700", "color": "#333",
                           "marginBottom": "2px"}),
            html.P("Avg Humidity",
                   style={"color": "#888", "margin": "0",
                          "fontSize": "13px"}),
        ]), style=card_style), width=3),
    ]

    desc = weather_description(temp, precip, wind, weathercode)
    weather_desc = (f"Historical average for {city} in "
                    f"{MONTH_NAMES[month]}: {desc}")

    label, confidence = predict_clothing(
        temp, precip, wind, humidity, month, season
    )
    details = CLOTHING_DETAILS.get(label, CLOTHING_DETAILS["Casual wear"])

    source_text = (
        f"Predicted by Random Forest Classifier using the "
        f"{MONTH_NAMES[month]} historical average"
    )

    rec_card = dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div(details["emoji"],
                             style={"fontSize": "52px",
                                    "textAlign": "center",
                                    "lineHeight": "80px",
                                    "width": "80px",
                                    "height": "80px",
                                    "margin": "auto"}),
                ], width=2,
                   className="d-flex align-items-center justify-content-center"),
                dbc.Col([
                    html.P("OUTFIT RECOMMENDATION",
                           style={"color": "#888", "margin": "0",
                                  "fontSize": "11px",
                                  "letterSpacing": "0.1em"}),
                    html.H2(label,
                            style={"color": details["color"],
                                   "fontWeight": "700",
                                   "marginBottom": "4px"}),
                    html.P(details["suggestion"],
                           style={"color": "#444", "fontSize": "15px",
                                  "marginBottom": "6px"}),
                    html.P(f"What to wear: {details['items']}",
                           style={"color": "#666", "fontSize": "13px",
                                  "marginBottom": "10px"}),
                    dbc.Progress(
                        value=confidence,
                        label=f"Model Confidence {confidence}%",
                        color="success",
                        style={"height": "22px",
                               "borderRadius": "10px",
                               "maxWidth": "350px"},
                    ),
                    html.P(source_text,
                           style={"color": "#aaa", "fontSize": "11px",
                                  "marginTop": "6px"}),
                ], width=10),
            ]),
        ])
    ], style={"borderRadius": "16px",
              "border": f"2px solid {details['color']}",
              "boxShadow": "0 4px 16px rgba(0,0,0,0.08)",
              "padding": "8px"})

    times = hourly.get("time", [])[:168:6]
    temps = hourly.get("temperature_2m", [])[:168:6]
    forecast_fig = go.Figure()
    forecast_fig.add_trace(go.Scatter(
        x=times, y=temps,
        mode="lines+markers",
        line=dict(color="#1D9E75", width=2),
        marker=dict(size=6, color="#1D9E75"),
        fill="tozeroy",
        fillcolor="rgba(29,158,117,0.08)",
        name="Temperature",
    ))
    forecast_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Temperature (C)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=10, b=40, l=40, r=10),
        height=240,
        font=dict(family="Segoe UI"),
    )
    forecast_fig.update_xaxes(showgrid=False)
    forecast_fig.update_yaxes(gridcolor="#f0f0f0")

    season_map = {0: "Spring", 1: "Summer", 2: "Autumn", 3: "Winter"}
    try:
        merged_docs = list(db[MONGO_COLLECTIONS["merged"]].find(
            {"city": city},
            {"season": 1, "review_count": 1, "_id": 0}
        ))
        merged_df = pd.DataFrame(merged_docs)
        if not merged_df.empty and "review_count" in merged_df.columns:
            merged_df["season_name"] = merged_df["season"].map(season_map)
            season_demand = merged_df.groupby(
                "season_name")["review_count"].mean().reset_index()
            demand_fig = px.bar(
                season_demand,
                x="season_name",
                y="review_count",
                color="season_name",
                color_discrete_map={
                    "Spring": "#A8D5A2",
                    "Summer": "#FFB347",
                    "Autumn": "#C1783C",
                    "Winter": "#7EC8E3",
                },
                labels={"review_count": "Avg Reviews",
                        "season_name": "Season"},
            )
        else:
            demand_fig = go.Figure()
    except Exception:
        demand_fig = go.Figure()

    demand_fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=10, b=40, l=40, r=10),
        height=240,
        showlegend=False,
        font=dict(family="Segoe UI"),
    )
    demand_fig.update_xaxes(showgrid=False)
    demand_fig.update_yaxes(gridcolor="#f0f0f0")

    base_demand = predict_demand(
        temp, precip, wind, humidity, month, season, city
    )
    demand_scores = get_demand_scores(base_demand, temp, precip, label)

    max_d = max(demand_scores) if max(demand_scores) > 0 else 1
    demand_levels = []
    bar_colors = []
    for d in demand_scores:
        pct = d / max_d
        if pct >= 0.70:
            demand_levels.append("HIGH")
            bar_colors.append("#1D9E75")
        elif pct >= 0.35:
            demand_levels.append("MEDIUM")
            bar_colors.append("#FFB347")
        else:
            demand_levels.append("LOW")
            bar_colors.append("#D3D1C7")

    forecast_demand_fig = go.Figure()
    forecast_demand_fig.add_trace(go.Bar(
        x=CATEGORIES,
        y=demand_scores,
        marker_color=bar_colors,
        text=demand_levels,
        textposition="outside",
        textfont=dict(size=12, family="Segoe UI"),
    ))
    forecast_demand_fig.update_layout(
        xaxis_title="Clothing Category",
        yaxis_title="Predicted Demand Score",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=30, b=80, l=40, r=10),
        height=340,
        font=dict(family="Segoe UI"),
    )
    forecast_demand_fig.update_xaxes(showgrid=False, tickangle=15)
    forecast_demand_fig.update_yaxes(gridcolor="#f0f0f0")

    try:
        hist_docs = list(db[MONGO_COLLECTIONS["merged"]].find(
            {"city": city},
            {"avg_temperature": 1, "clothing_category": 1,
             "month": 1, "year": 1, "_id": 0}
        ))
        hist_df = pd.DataFrame(hist_docs)
        if not hist_df.empty:
            hist_fig = px.scatter(
                hist_df,
                x="avg_temperature",
                y="month",
                color="clothing_category",
                size_max=10,
                labels={
                    "avg_temperature": "Avg Temperature (C)",
                    "month": "Month",
                    "clothing_category": "Category",
                },
                color_discrete_map={
                    "Light clothing":      "#FF8C00",
                    "Casual wear":         "#1D9E75",
                    "Layered clothing":    "#5B8DB8",
                    "Heavy outerwear":     "#7B5EA7",
                    "Waterproof clothing": "#2E86AB",
                    "Thermal wear":        "#4A90D9",
                },
            )
        else:
            hist_fig = go.Figure()
    except Exception:
        hist_fig = go.Figure()

    hist_fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=10, b=40, l=40, r=10),
        height=320,
        font=dict(family="Segoe UI"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    hist_fig.update_xaxes(showgrid=False)
    hist_fig.update_yaxes(gridcolor="#f0f0f0")

    return (clock_display, live_strip, weather_cards, weather_desc, rec_card,
            forecast_fig, demand_fig,
            forecast_demand_fig, hist_fig)


if __name__ == "__main__":
    app.run(
        host=DASH_HOST,
        port=DASH_PORT,
        debug=DASH_DEBUG,
    )