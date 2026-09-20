"""
FastAPI application for the Flight Price Prediction API.

The model is loaded once at startup via a lifespan context manager and shared
across all requests. Each request gets a unique correlation ID via middleware.
Inputs are validated with Pydantic before any prediction code runs.
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from flight_predictor.config import API_TITLE, API_VERSION, MODEL_INFO_PATH
from flight_predictor.logging_conf import (
    CorrelationIDMiddleware,
    get_correlation_id,
    setup_logging,
)
from flight_predictor.predict import BaseFlightPredictor, load_predictor

logger = logging.getLogger(__name__)

# Shared predictor instance — loaded once at startup, reused for every request
app_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    setup_logging(
        level=os.getenv("LOG_LEVEL", "INFO"), fmt=os.getenv("LOG_FORMAT", "json")
    )
    logger.info("Flight Price starting up ...")

    if app_state.get("predictor") is None:
        use_onnx = os.getenv("FLIGHT_USE_ONNX", "false").lower() == "true"
        try:
            predictor = load_predictor(use_onnx=use_onnx)
            app_state["predictor"] = predictor
            logger.info(
                "Predictor loaded successfully",
                extra={"predictor_type": predictor.__class__.__name__},
            )
        except FileNotFoundError as e:
            logger.error(f"Failed to load predictor: {e}")
            # App starts in degraded mode; /predict returns 503 until model is available
            app_state["predictor"] = None
            app_state["load_error"] = str(e)

    yield

    # Shutdown
    logger.info("Flight Price API shutting down ...")
    app_state.clear()


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="""
    Predicts Indian domestic flight prices in INR using XGBoost.

    ## Usage
    POST `/predict` with flight details to get a price estimate.

    ## Model
    XGBoost with log1p target transformation (TransformedTargetRegressor).
    Baseline R² = 0.7598. Target R² > 0.80 after bug fixes.
    """,
    lifespan=lifespan,
)

app.add_middleware(CorrelationIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=None, tags=["Infrastructure"])
async def health_check():
    predictor = app_state.get("predictor")
    if predictor is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "version": API_VERSION,
            },
        )
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "model_loaded": True,
            "predictor_type": predictor.__class__.__name__,
            "version": API_VERSION,
        },
    )


@app.get("/model-info", tags=["Infrastructure"])
async def model_info():
    if not MODEL_INFO_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="model_info.json not found. Run `flight-train` first.",
        )

    with open(MODEL_INFO_PATH) as f:
        content = f.read().replace(": NaN", ": null").replace(": nan", ": null")
        info = json.loads(content)

    return info


@app.post("/predict", tags=["Prediction"])
async def predict(request: Request):
    from api.schemas import FlightPredictionRequest, FlightPredictionResponse

    # Parse and validate request body
    body = await request.json()
    try:
        flight_request = FlightPredictionRequest(**body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    # check predictor is loaded
    predictor: BaseFlightPredictor | None = app_state.get("predictor")
    if predictor is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Check /health for details."
        )

    # Convert request to DataFrame and predict
    input_df = flight_request.to_dataframe()

    try:
        result = predictor.predict_safe(input_df)
    except RuntimeError as e:
        logger.error("Prediction error", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(
        "Prediction served.",
        extra={
            "predicted_price_INR": result["predicted_price_inr"],
            "airline": flight_request.airline.value,
            "source": flight_request.source.value,
            "destination": flight_request.destination.value,
            "duration": flight_request.duration,
            "stops": flight_request.total_stops.value,
        },
    )

    return FlightPredictionResponse(**result, request_id=get_correlation_id())


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(request: Request):
    from api.schemas import FlightPredictionRequest

    body = await request.json()
    if not isinstance(body, list):
        raise HTTPException(status_code=422, detail="Body must be a JSON array")

    predictor: BaseFlightPredictor | None = app_state.get("predictor")
    if predictor is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Check /health for details."
        )

    results = []
    for i, item in enumerate(body):
        try:
            flight_request = FlightPredictionRequest(**item)
            result = predictor.predict_safe(flight_request.to_dataframe())
            results.append({"index": i, **result})
        except Exception as e:
            results.append({"index": i, "status": "error", "error": str(e)})

    return {"predictions": results, "total": len(results)}
