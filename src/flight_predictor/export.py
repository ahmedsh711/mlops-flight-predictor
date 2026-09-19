import logging
from pathlib import Path

import joblib
import numpy as np

from flight_predictor.config import MODELS_DIR, ONNX_MODEL_PATH, PIPELINE_PATH
from flight_predictor.data import load_raw_data
from flight_predictor.features import engineer_features, get_feature_columns

logger = logging.getLogger(__name__)


def export_to_onnx(
    pipeline_path: Path | None = None, onnx_output_path: Path | None = None
) -> Path:
    try:
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError:
        raise ImportError("onnxmltools not installed. Run: uv pip install onnxmltools")

    pipeline_path = pipeline_path or PIPELINE_PATH
    onnx_output_path = onnx_output_path or ONNX_MODEL_PATH

    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"pipeline not found at {pipeline_path}.Run `flight-train` first."
        )

    # Load trained sklearn pipeline
    pipeline = joblib.load(pipeline_path)

    # Extract inner XGBRegressor from Pipeline -> TTR -> XGBRegressor
    ttr = pipeline.named_steps["model"]  # TransformedTargetRegressor
    xgb_model = ttr.regressor_

    # Determine input shape from preprocessor output
    preprocessor = pipeline.named_steps["preprocessor"]

    # Load a sample to get feature dimensions
    df_raw = load_raw_data()
    df = engineer_features(df_raw)
    feature_cols = get_feature_columns()
    X_sample = df[feature_cols].head(10)
    X_transformed = preprocessor.transform(X_sample)
    n_features = X_transformed.shape[1]

    logger.info("Exporting XGBoost to ONNX", extra={"n_features": n_features})

    # convert XGBoost -> ONNX
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(xgb_model, initial_types=initial_type)

    # Save ONNX MODEL
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(onnx_output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Save preprocessor separately for onnx_predictor (if not already saved by training)
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    if not preprocessor_path.exists():
        joblib.dump(preprocessor, preprocessor_path)

    logger.info(
        "ONNX export complete",
        extra={
            "onnx_path": str(onnx_output_path),
            "preprocessor_path": str(preprocessor_path),
            "n_features": n_features,
        },
    )
    return onnx_output_path


def verify_onnx_parity(pipeline_path: Path | None = None, atol: float = 0.5) -> bool:
    """
    Verify ONNX predictions match sklearn predictions within tolerance.
    Returns True if parity is confirmed, raises AssertionError otherwise.
    """
    import onnxruntime as rt

    pipeline_path = pipeline_path or PIPELINE_PATH
    pipeline = joblib.load(pipeline_path)

    df_raw = load_raw_data()
    df = engineer_features(df_raw)
    feature_cols = get_feature_columns()
    X = df[feature_cols].head(50)

    # Sklearn predictions
    sklearn_preds = pipeline.predict(X)

    # Onnx predictions
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X).astype(np.float32)

    session = rt.InferenceSession(
        str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    log_preds = session.run(None, {input_name: X_transformed})[0].flatten()
    onnx_preds = np.expm1(log_preds)

    max_diff = np.abs(sklearn_preds - onnx_preds).max()
    if max_diff > atol:
        raise AssertionError(
            f"ONNX parity check failed! Max diff = {max_diff:.6f} > {atol}"
        )

    logger.info(
        "ONNX parity verified", extra={"max_diff": float(max_diff), "atol": atol}
    )
    return True


def main() -> None:
    from flight_predictor.logging_conf import setup_logging

    setup_logging()
    path = export_to_onnx()
    print(f"\n ONNX model exported to: {path}")
    verify_onnx_parity()
    print("Onnx parity verified (Sklearn ~ ONNX Predictions)")


if __name__ == "__main__":
    main()
