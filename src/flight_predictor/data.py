import logging
from pathlib import Path

import pandas as pd

from flight_predictor.config import (
    DATA_RAW_DIR,
    MAX_PRICE_INR,
    MIN_PRICE_INR,
    PRICE_COLUMN,
    RARE_AIRLINES,
)

logger = logging.getLogger(__name__)

# Required columns from the raw csv
REQUIRED_COLUMNS = [
    "Airline",
    "Date_of_Journey",
    "Source",
    "Destination",
    "Route",
    "Dep_Time",
    "Arrival_Time",
    "Duration",
    "Total_Stops",
    "Additional_Info",
    "Price",
]


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """
    Load raw flight data CSV and perform basic validation.

    Args:
        path: Path to CSV file. Defaults to DATA_RAW_DIR / 'flight_data.csv'

    Returns:
        DataFrame with validated raw data.

    Raises:
        FileNotFoundError: If CSV doesn't exist.
        ValueError: If required columns are missing.
    """
    if path is None:
        path = DATA_RAW_DIR / "flight_data.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Flight data not found at {path}. Copy your CSV to data/raw/flight_data.csv"
        )

    logger.info("Loading flight data", extra={"path": str(path)})
    df = pd.read_csv(path)

    # Validate required columns
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. Found: {list(df.columns)}"
        )

    # Basic shape validation
    if len(df) < 100:
        raise ValueError(
            f"Dataset too small: {len(df)} rows. "
            "Need at least 100 rows for meaningful training."
        )

    # Map rare airlines to "Other" before returning raw data
    df["Airline"] = df["Airline"].apply(lambda a: "Other" if a in RARE_AIRLINES else a)

    # Remove obvious price outliers
    if PRICE_COLUMN in df.columns:
        n_before = len(df)

        df = df[
            (df[PRICE_COLUMN] >= MIN_PRICE_INR) & (df[PRICE_COLUMN] <= MAX_PRICE_INR)
        ]

        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.warning(
                "Dropped price outliers",
                extra={"n_dropped": n_dropped, "remaining": len(df)},
            )

    logger.info(
        "Data loaded successfully",
        extra={"n_rows": len(df), "n_columns": len(df.columns)},
    )

    return df
