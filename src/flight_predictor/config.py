import os
from pathlib import Path

# Project Root
# __file__ = .../src/flight_predictor/config.py
# .parent = .../src/flight_predictor/
# .parent = .../src/
# .parent = .../mlops-flight-predictor/

ROOT_DIR = Path(__file__).parent.parent.parent

# Data Paths:
DATA_DIR = ROOT_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROC_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

# Create directories if they don't exist:
for _dir in [DATA_RAW_DIR, DATA_PROC_DIR, MODELS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# Models Artifact Paths:
PIPELINE_PATH = MODELS_DIR / "pipeline.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
ONNX_MODEL_PATH = MODELS_DIR / "xgb_model.onnx"
MODEL_INFO_PATH = MODELS_DIR / "model_info.json"
PREV_BEST_PATH = MODELS_DIR / "previous_best_r2.txt"

# Reproducity:
RANDOM_SEED = 42

# Training & Quality Gate:
TEST_SIZE = 0.2  # 80 20 TRAIN TEST SPLIT
CV_FOLDS = 5
R2_THRESHOLD = float(os.getenv("R2_THRESHOLD", "0.80"))
MAX_REGRESSION_ALLOWED = float(os.getenv("MAX_REGRESSION_ALLOWED", "0.02"))
MAX_PRICE_INR = 100_000  # SANITY CHECK
MIN_PRICE_INR = 500  # SANITY CHECK

# FEATURE ENGINEERING:
DURATION_COLUMN = "Duration"
PRICE_COLUMN = "Price"
DATE_COLUMN = ["Date_of_Journey", "Dep_Time", "Arrival_Time"]

# Indian airports (hardcoded — coordinates don't change)
AIRPORT_CODES = ["BLR", "CCU", "DEL", "MAA", "BOM", "COK", "HYD"]
AIRPORT_COORDS: dict[str, tuple[float, float]] = {
    "BLR": (12.9716, 77.5946),  # Kempegowda, Bengaluru
    "DEL": (28.7041, 77.1025),  # Indira Gandhi, Delhi
    "BOM": (19.0760, 72.8777),  # Chhatrapati Shivaji, Mumbai
    "MAA": (13.0827, 80.2707),  # Chennai International
    "CCU": (22.5726, 88.3639),  # Netaji Subhas, Kolkata
    "COK": (9.9312, 76.2673),  # Cochin International
    "HYD": (17.3850, 78.4867),  # Rajiv Gandhi, Hyderabad
}

# Airlines (ordered by frequency in training data)
KNOWN_AIRLINES = [
    "IndiGo",
    "Air India",
    "Jet Airways",
    "SpiceJet",
    "GoAir",
    "Vistara",
    "Air Asia",
]
# Rare airlines mapped to 'Other' during preprocessing
RARE_AIRLINES = [
    "Multiple carriers Premium economy",
    "Jet Airways Business",
    "Vistara Premium economy",
    "Trujet",
]

# API:
API_TITLE = "Flight Price Prediction API"
API_VERSION = "1.0.0"
API_HOST = "8000"

# LOGGING:
LOG_LEVEL = "INFO"
LOG_FORMAT = "json"
