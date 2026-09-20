"""
Integration tests for the FastAPI endpoints.
Uses a mocked predictor — tests API routing and validation logic,
not model accuracy (that's covered in test_serialization.py).
"""

VALID_PAYLOAD = {
    "airline": "IndiGo",
    "date_of_journey": "15/06/2024",
    "source": "Delhi",
    "destination": "Bangalore",
    "dep_time": "06:00",
    "arrival_time": "08:45",
    "duration": "2h 45m",
    "total_stops": "non-stop",
    "additional_info": "No info",
}


class TestHealthEndpoint:
    def test_health_returns_200_when_model_loaded(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_return_503_when_model_not_loaded(self, test_client):
        from api.main import app_state

        original = app_state.pop("predictor", None)
        try:
            response = test_client.get("/health")
            assert response.status_code == 503
        finally:
            if original is not None:
                app_state["predictor"] = original


class TestPredictorEndpoint:
    def test_valid_request_returns_200(self, test_client):
        response = test_client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_response_contains_price(self, test_client):
        response = test_client.post("/predict", json=VALID_PAYLOAD)
        data = response.json()
        assert "predicted_price_inr" in data
        assert data["predicted_price_inr"] == 5500.0  # from mock

    def test_response_contains_request_id(self, test_client):
        response = test_client.post("/predict", json=VALID_PAYLOAD)
        data = response.json()
        assert "request_id" in data
        assert len(data["request_id"]) > 0

    def test_missing_required_field_returns_422(self, test_client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "airline"}
        response = test_client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_airline_returns_422(self, test_client):
        payload = {**VALID_PAYLOAD, "airline": "FlyingCarpet Airways"}
        response = test_client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_same_source_and_destination_returns_422(self, test_client):
        payload = {**VALID_PAYLOAD, "destination": "Delhi"}
        response = test_client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_city_returns_422(self, test_client):
        payload = {**VALID_PAYLOAD, "source": "Atlantis"}
        response = test_client.post("/predict", json=payload)
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    def test_batch_predict_multiple_flight(self, test_client):
        payload = [VALID_PAYLOAD, {**VALID_PAYLOAD, "airline": "Air India"}]
        response = test_client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_rejects_non_list(self, test_client):
        response = test_client.post("/predict/batch", json=VALID_PAYLOAD)
        assert response.status_code == 422


class TestPredictorClasses:
    def test_sklearn_predictor(self, sample_df):
        from flight_predictor.predict import SklearnPredictor

        predictor = SklearnPredictor()
        price = predictor.predict(sample_df)
        assert isinstance(price, float)
        assert price > 0

        safe_res = predictor.predict_safe(sample_df)
        assert safe_res["status"] == "success"
        assert "predicted_price_inr" in safe_res

    def test_onnx_predictor(self, sample_df):
        from flight_predictor.predict import OnnxPredictor

        predictor = OnnxPredictor()
        price = predictor.predict(sample_df)
        assert isinstance(price, float)
        assert price > 0

        safe_res = predictor.predict_safe(sample_df)
        assert safe_res["status"] == "success"

    def test_load_predictor_factory(self):
        from flight_predictor.predict import (
            OnnxPredictor,
            SklearnPredictor,
            load_predictor,
        )

        p_sklearn = load_predictor(use_onnx=False)
        assert isinstance(p_sklearn, SklearnPredictor)

        p_onnx = load_predictor(use_onnx=True)
        assert isinstance(p_onnx, OnnxPredictor)


class TestDataAndTrain:
    def test_load_raw_data(self):
        from flight_predictor.data import load_raw_data

        df = load_raw_data()
        assert len(df) >= 100
        assert "Price" in df.columns

    def test_load_raw_data_invalid_path(self, tmp_path):
        import pytest

        from flight_predictor.data import load_raw_data

        with pytest.raises(FileNotFoundError):
            load_raw_data(tmp_path / "non_existent.csv")

    def test_build_pipeline(self):
        from flight_predictor.train import build_pipeline

        pipe = build_pipeline()
        assert "preprocessor" in pipe.named_steps
        assert "model" in pipe.named_steps

    def test_evaluate_model(self, sample_batch_df):
        from flight_predictor.features import engineer_features, get_feature_columns
        from flight_predictor.train import build_pipeline, evaluate_model

        df = engineer_features(sample_batch_df)
        X = df[get_feature_columns()]
        y = df["Price"].astype(float)

        pipe = build_pipeline({"n_estimators": 5, "max_depth": 2})
        pipe.fit(X, y)
        metrics = evaluate_model(pipe, X, y)
        assert "r2_inr_space" in metrics
        assert "rmse_inr" in metrics
        assert "mae_inr" in metrics

    def test_verify_onnx_parity(self):
        from flight_predictor.export import verify_onnx_parity

        assert verify_onnx_parity() is True
