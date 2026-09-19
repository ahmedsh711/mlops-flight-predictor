"""
Serialization tests — verify ONNX and sklearn produce matching predictions.
Also includes a latency benchmark.

These tests require trained artifacts (pipeline.joblib, xgb_model.onnx).
They are skipped automatically if artifacts don't exist yet.
"""

import time

import numpy as np
import pytest

from flight_predictor.config import ONNX_MODEL_PATH, PIPELINE_PATH

# skip if artifacts don't exist
pytestmark = pytest.mark.skipif(
    not PIPELINE_PATH.exists(),
    reason="Trained model artifacts not found. Run `flight-train` and `flight-export` first.",
)


@pytest.fixture(scope="module")
def sample_engineered_features():
    """Load a sample of features from the training data for parity testing."""
    import pytest

    from flight_predictor.data import load_raw_data
    from flight_predictor.features import engineer_features, get_feature_columns

    try:
        df_raw = load_raw_data()
        df = engineer_features(df_raw)
        return df[get_feature_columns()].head(50)
    except FileNotFoundError:
        pytest.skip("Training data not avaliable")


class TestOnnxParity:
    """Verify ONNX predictions match sklearn pipeline predictions."""

    @pytest.mark.skipif(
        not ONNX_MODEL_PATH.exists(),
        reason="ONNX model not found. Run `flight-export` first.",
    )
    def test_onnx_sklearn_predictions_match(
        self, sklearn_pipeline, sample_engineered_features
    ):
        import joblib
        import onnxruntime as rt

        from flight_predictor.config import MODELS_DIR

        X = sample_engineered_features

        # sklearn predictions (in INR — TTR applies expm1 automatically)
        sklearn_preds = sklearn_pipeline.predict(X)

        # ONNX predictions (manual expm1 needed)
        preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
        X_transformed = preprocessor.transform(X).astype(np.float32)

        session = rt.InferenceSession(
            str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        log_preds = session.run(None, {input_name: X_transformed})[0].flatten()
        onnx_preds = np.expm1(log_preds)

        max_diff = np.abs(sklearn_preds - onnx_preds).max()
        mean_diff = np.abs(sklearn_preds - onnx_preds).mean()

        assert max_diff <= 0.5, (
            f"ONNX predictions differ from sklearn by up to {max_diff:.4f} INR."
            f"Mean difference: {mean_diff:.4f} INR."
            f"Check that expm1 is applied correctly in OnnxPredictor."
        )

    def test_predictions_in_realistic_inr_range(
        self, sklearn_pipeline, sample_engineered_features
    ):
        preds = sklearn_pipeline.predict(sample_engineered_features)

        assert preds.min() >= 500, (
            f"Some predictions below 500 INR (min={preds.min():.2f})."
            "Prices in log space would look like 7-8 here."
        )
        assert preds.max() <= 100_000, (
            f"Some predictions above 100,000 INR (max={preds.max():.2f})."
            "Possible extreme outlier or model issue."
        )


class TestModelPerformance:
    def test_r2_above_threshold(self, sklearn_pipeline):
        import json

        from flight_predictor.config import MODEL_INFO_PATH, R2_THRESHOLD

        if not MODEL_INFO_PATH.exists():
            pytest.skip("model_info.json not found")

        with open(MODEL_INFO_PATH) as f:
            info = json.load(f)

        r2 = info["metrics"]["r2_inr_space"]
        assert r2 >= R2_THRESHOLD, (
            f"Model R2= {r2:.4f} is below threshold {R2_THRESHOLD}."
            "Review feature engineering and hyperparameters."
        )


class TestInferenceLatency:
    def test_sklearn_inference_under_100ms(
        self, sklearn_pipeline, sample_engineered_features
    ):
        X = sample_engineered_features.head(1)

        # Warmup
        sklearn_pipeline.predict(X)

        # Measure
        n_runs = 100
        start = time.perf_counter()
        for _ in range(n_runs):
            sklearn_pipeline.predict(X)
        elapsed = (time.perf_counter() - start) / n_runs * 1000

        assert elapsed < 100, (
            f"Sklearn inference took {elapsed:.1f}ms per prediction."
            "Target is <100 ms for responsive API."
        )
