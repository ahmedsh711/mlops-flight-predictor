from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from flight_predictor.config import MODELS_DIR, ONNX_MODEL_PATH, PIPELINE_PATH
from flight_predictor.features import engineer_features, get_feature_columns

logger = logging.getLogger(__name__)


# Base class — both sklearn and ONNX predictors implement this contract
class BaseFlightPredictor(ABC):
    @abstractmethod
    def predict(self, input_df: pd.DataFrame) -> float: ...

    def predict_safe(self, input_df: pd.DataFrame) -> dict:
        try:
            price = self.predict(input_df)
            return {
                "predicted_price_inr": round(float(price), 2),
                "predictor_type": self.__class__.__name__,
                "status": "success",
            }
        except Exception as e:
            logger.error("Prediction failed", exc_info=True)
            raise RuntimeError(f"Prediction failed: {e}") from e


class SklearnPredictor(BaseFlightPredictor):
    def __init__(self, model_path: Path | None = None) -> None:
        path = model_path or PIPELINE_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Pipeline not found at {path}. Run `flight-train` first."
            )
        self._pipeline = joblib.load(path)
        logger.info("Sklearn pipeline loaded", extra={"path": str(path)})

    def predict(self, input_df: pd.DataFrame) -> float:
        features = engineer_features(input_df)
        feature_cols = get_feature_columns()
        X = features[feature_cols]
        price = self._pipeline.predict(X)
        return float(price[0])


class OnnxPredictor(BaseFlightPredictor):
    def __init__(
        self, onnx_path: Path | None = None, preprocessor_path: Path | None = None
    ) -> None:
        import onnxruntime as rt

        onnx_path = onnx_path or ONNX_MODEL_PATH
        preprocessor_path = preprocessor_path or (MODELS_DIR / "preprocessor.joblib")

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_path}. Run `flight-export` first."
            )

        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}.")

        self._session = rt.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )

        self._preprocessor = joblib.load(preprocessor_path)
        self._input_name = self._session.get_inputs()[0].name
        logger.info("ONNX predictor loaded", extra={"path": str(onnx_path)})

    def predict(self, input_df: pd.DataFrame) -> float:
        features = engineer_features(input_df)
        feature_cols = get_feature_columns()
        X = features[feature_cols]

        # apply sklearn preprocessor: OHE + scaling
        X_transformed = self._preprocessor.transform(X).astype(np.float32)

        # ONNX returns log-scale price — invert with expm1
        outputs = self._session.run(None, {self._input_name: X_transformed})

        log_price = outputs[0].flatten()[0]

        return float(np.expm1(log_price))


# Factory Function for FastAPI startup
def load_predictor(use_onnx: bool = False) -> BaseFlightPredictor:
    """
    Factory that returns the appropriate predictor.

    Args:
        use_onnx: If True, load OnnxPredictor. Otherwise load SklearnPredictor.
                  Controlled by FLIGHT_USE_ONNX env var in production.

    Returns:
        A concrete BaseFlightPredictor instance.
    """
    if use_onnx:
        return OnnxPredictor()
    return SklearnPredictor()


def main() -> None:
    """Quick Smoke Test"""

    from flight_predictor.logging_conf import setup_logging

    setup_logging()

    # Sample flight: IndiGo, Delhi to Bangalore, 1 stop, 2h 45m
    sample = pd.DataFrame(
        [
            {
                "Airline": "IndiGo",
                "Date_of_Journey": "15/06/2024",
                "Source": "Delhi",
                "Destination": "Bangalore",
                "Route": "DEL → BLR",
                "Dep_Time": "06:00",
                "Arrival_Time": "08:45",
                "Duration": "2h 45m",
                "Total_Stops": "non-stop",
                "Additional_Info": "No info",
            }
        ]
    )

    import os

    use_onnx = os.getenv("FLIGHT_USE_ONNX", "false").lower() == "true"
    predictor = load_predictor(use_onnx=use_onnx)
    result = predictor.predict_safe(sample)
    print(f"\n Sample prediction: {result['predicted_price_inr']:,.0f} INR")


if __name__ == "__main__":
    main()
