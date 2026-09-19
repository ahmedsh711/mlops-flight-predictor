import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from flight_predictor.config import AIRPORT_COORDS, KNOWN_AIRLINES


def parse_duration_to_minutes(duration: str) -> int:
    s = str(duration).strip()

    # handle already numeric values
    if s.isdigit():
        return int(s)

    hours_match = re.search(r"(\d+)h", s)
    mins_match = re.search(r"(\d+)m", s)

    h = int(hours_match.group(1)) if hours_match else 0
    m = int(mins_match.group(1)) if mins_match else 0

    if h == 0 and m == 0:
        # Unkown format - return 0 and let the model handle it
        return 0

    return h * 60 + m


def parse_stops(stops: str) -> int:
    s = str(stops).lower().strip()
    if "non" in s:
        return 0
    match = re.search(r"(\d+)", s)
    return int(match.group(1)) if match else 0


def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from date/time columns.

    Adds:
        journey_day, journey_month, journey_day_of_week
        dep_hour, dep_minute
        arrival_hour, arrival_minute

    These are important because:
    - Prices spike during holidays and weekends
    - Morning and evening flights are often more expensive
    - Monsoon season (June-September) affects routes and prices
    """
    df = df.copy()

    # Journey date features
    journey_dt = pd.to_datetime(df["Date_of_Journey"], dayfirst=True, errors="coerce")
    df["journey_day"] = journey_dt.dt.day
    df["journey_month"] = journey_dt.dt.month
    df["journey_day_of_week"] = journey_dt.dt.dayofweek  # 0= Mon, 6=Sun

    # Departure time features
    dep_dt = pd.to_datetime(df["Dep_Time"], format="%H:%M", errors="coerce")
    df["dep_hour"] = dep_dt.dt.hour
    df["dep_minute"] = dep_dt.dt.minute

    # Arrival time features
    arr_str = df["Arrival_Time"].astype(str).str.split().str[0]
    arr_dt = pd.to_datetime(arr_str, format="%H:%M", errors="coerce")
    df["arrival_hour"] = arr_dt.dt.hour
    df["arrival_minute"] = arr_dt.dt.minute

    return df


CITY_TO_AIRPORT: dict[str, str] = {
    "BANGALORE": "BLR",
    "BANGLORE": "BLR",
    "DELHI": "DEL",
    "NEW DELHI": "DEL",
    "MUMBAI": "BOM",
    "CHENNAI": "MAA",
    "KOLKATA": "CCU",
    "COCHIN": "COK",
    "HYDERABAD": "HYD",
}


def get_airport_coords(code: str) -> tuple[float, float]:
    """
    Return (latitude, longitude) for an airport code or city name.
    Falls back to (0.0, 0.0) if code is unknown — does NOT call any API.
    """
    key = str(code).strip().upper()
    key = CITY_TO_AIRPORT.get(key, key)
    return AIRPORT_COORDS.get(key, (0.0, 0.0))


def compute_route_distance(source: str, dest: str) -> float:
    """
    Compute approximate great-circle distance between two airports in km.
    Uses Haversine formula — no external API calls.
    """
    import math

    src_lat, src_lon = get_airport_coords(source)
    dst_lat, dst_lon = get_airport_coords(dest)

    # Haversine Formula:
    R = 6371.0  # Earth radius in KM
    phi1, phi2 = math.radians(src_lat), math.radians(dst_lat)
    d_phi1 = math.radians(dst_lat - src_lat)
    d_lam = math.radians(dst_lon - src_lon)

    a = (
        math.sin(d_phi1 / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    )

    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


## DataFrame transformation (applies all pure functions)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations to raw flight DataFrame.
    Returns a new DataFrame with engineered features. Does NOT modify original.

    This function is called:
    1. During training: on the training + validation DataFrames
    2. During prediction: on the single-row inference DataFrame
    The exact same transformations must be applied in both cases.
    """
    df = df.copy()

    df["duration_minutes"] = df["Duration"].apply(parse_duration_to_minutes)
    df["n_stops"] = df["Total_Stops"].apply(parse_stops)
    df = extract_date_features(df)
    df["route_distance_km"] = df.apply(
        lambda r: compute_route_distance(r["Source"], r["Destination"]), axis=1
    )

    df["avg_speed_kmh"] = np.where(
        df["duration_minutes"] > 0,
        df["route_distance_km"] / (df["duration_minutes"] / 60),
        0.0,
    )

    df["airline_clean"] = df["Airline"].where(
        df["Airline"].isin(KNOWN_AIRLINES), other="Other"
    )

    return df


## scikit-learn ColumnTransformer - handle encoding for model
# Features used by the model (in order):
NUMERIC_FEATURES = [
    "duration_minutes",
    "n_stops",
    "route_distance_km",
    "avg_speed_kmh",
    "journey_day",
    "journey_month",
    "journey_day_of_week",
    "dep_hour",
    "dep_minute",
    "arrival_hour",
    "arrival_minute",
]

CATEGORICAL_FEATURES = ["airline_clean", "Source", "Destination"]


def build_preprocessor() -> ColumnTransformer:
    """
    Numeric features: StandardScaler (zero mean, unit variance)
    Categorical features: OneHotEncoder (sparse_output=False for XGBoost)

    The preprocessor is fit ONLY on training data (no data leakage).
    It is saved to disk alongside the model so prediction uses
    the exact same scaling parameters.
    """
    numeric_transformer = Pipeline([("scaler", StandardScaler())])

    categorical_transformer = Pipeline(
        [("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def get_feature_columns() -> list[str]:
    """Return list of features that build_preprocessor expects"""
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES
