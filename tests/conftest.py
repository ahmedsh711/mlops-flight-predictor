import numpy as np
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from unittest.mock import MagicMock, patch

from flight_predictor.features import build_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES

# Sample flight data fixtures
@pytest.fixture
def sample_raw_flight():
    """A single raw flight row — as it would come from the CSV."""
    return {
        "Airline":         "IndiGo",
        "Date_of_Journey": "15/06/2024",
        "Source":          "Delhi",
        "Destination":     "Bangalore",
        "Route":           "DEL → BLR",
        "Dep_Time":        "06:00",
        "Arrival_Time":    "08:45",
        "Duration":        "2h 45m",
        "Total_Stops":     "non-stop",
        "Additional_Info": "No info",
        "Price":           5500.0,
    }

@pytest.fixture
def sample_batch_df(sample_raw_flight):
    """5-row DataFrame covering edge cases."""
    rows = [
        sample_raw_flight,  # normal IndiGo Delhi→Bangalore
        {**sample_raw_flight, "Airline": "Air India", "Price": 12000.0},
        {**sample_raw_flight, "Duration": "45m", "Total_Stops": "non-stop", "Price": 3200.0},  # short flight — Bug #3
        {**sample_raw_flight, "Source": "Mumbai", "Destination": "Chennai", "Price": 7500.0},
        {**sample_raw_flight, "Total_Stops": "2 stops", "Duration": "8h 30m", "Price": 18000.0},
    ]
    return pd.DataFrame(rows)

@pytest.fixture
def sample_df(sample_raw_flight):
    """Single-row DataFrame fixture."""
    return pd.DataFrame([sample_raw_flight])

@pytest.fixture(scope="module")
def sklearn_pipeline():
    """Trained sklearn pipeline fixture."""
    import joblib
    from flight_predictor.config import PIPELINE_PATH
    if not PIPELINE_PATH.exists():
        pytest.skip("Pipeline artifact not found")
    return joblib.load(PIPELINE_PATH)

## Mock predictor fixture
@pytest.fixture
def mock_predictor():
    predictor = MagicMock()
    predictor.__class__.__name__ = "MockPredictor"
    predictor.predict.return_value = 5500.0
    predictor.predict_safe.return_value = {
        "predicted_price_inr": 5500.0,
        "predictor_type": "MockPredictor",
        "status": "success"
    }
    return predictor

## FastAPI test client fixture
@pytest.fixture
def test_client(mock_predictor):
    """
    FastAPI TestClient with the real app but a mocked predictor.
    The lifespan is bypassed ; we inject the mock directly.
    """
    from fastapi.testclient import TestClient
    from api.main import app, app_state

    # Inject Mock Predictor directly into app_state
    app_state["predictor"] = mock_predictor

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client

    # Clean Up after test
    app_state.pop("predictor", None)