# Module 1 Report — Flight Price Prediction
**Student:** Ahmed Al-Shobaki  
**Date:** September 20, 2026  
**Project:** Indian Domestic Flight Price Prediction (XGBoost · INR · R² target > 0.80)

---

## Executive Summary

Starting with a monolithic Jupyter notebook (`extract.py`) containing five critical bugs and an R² of 0.7598, I refactored the project into a modular, production-ready MLOps Level 1 system. The packaged Python package achieves $R^2 = 0.8449$, $\text{RMSE} = ₹1,812$, and $\text{MAE} = ₹1,144$ on the held-out test set (comfortably exceeding the 0.80 threshold). The solution serves predictions via a high-performance FastAPI REST API, supports dual execution engines (Scikit-Learn Pipeline and ONNX Runtime), and runs reproducibly inside a multi-stage Docker container.

---

## 1. Bug Report

### Bug #1 — Hardcoded Windows Paths
**Location:** `extract.py` data ingestion  
**Root Cause:** `pd.read_csv(r"C:\00-Shobaki\...")` assumed a rigid Windows local directory structure.  
**Failure Mode:** `FileNotFoundError` on any developer environment other than the original author's, failing immediately in Linux/WSL2 and cloud CI/CD runners.  
**Fix:** Introduced dynamic anchor-relative resolution in `src/flight_predictor/config.py` using `pathlib.Path(__file__).resolve().parent.parent.parent / "data/raw/flight_data.csv"`. Resolves deterministically across Linux, macOS, and Windows runners without manual path manipulation.

### Bug #2 — Live Geopy API Calls
**Location:** `extract.py` coordinate extraction  
**Root Cause:** `Nominatim().geocode(city)` dispatched live HTTP requests to OpenStreetMap's geocoding API for every record during training.  
**Failure Mode 1:** `IndentationError` caused by leading whitespace inside `geocode( city)`.  
**Failure Mode 2:** `GeocoderTimedOut` and HTTP 429 when OpenStreetMap throttled requests under training loop volume.  
**Failure Mode 3:** Non-deterministic coordinates if OpenStreetMap database records were updated or re-ordered between training runs.  
**Fix:** Created an immutable dictionary constant `AIRPORT_COORDS` in `config.py` containing static, authoritative coordinates for all seven project airports (Delhi, Mumbai, Bangalore, Kolkata, Hyderabad, Chennai, Cochin). Replaces external network round-trips with an $O(1)$ in-memory hash lookup.

### Bug #3 — Duration Parser IndexError
**Location:** `extract.py` duration parsing function  
**Root Cause:** Naive string splitting `x.split("h")[1]` assumed every flight duration string contains an hour token `"h"`.  
**Failure Mode:** `IndexError: list index out of range` on short flights containing only minutes (e.g., `"45m"` or `"5m"`).  
**Affected Rows:** Row 6474 (`Air India`, Mumbai $\rightarrow$ Hyderabad, `"5m"`) and similar short-haul routes.  
**Fix:** Replaced naive slicing with regular expressions: `re.search(r"(\d+)h", s)` and `re.search(r"(\d+)m", s)`. Safely extracts hour and minute integers when present and defaults missing components to 0.

### Bug #4 — Missing TransformedTargetRegressor in Saved Model
**Location:** `extract.py` model evaluation vs model serialization  
**Root Cause:** `TransformedTargetRegressor` (with `np.log1p` forward transformation and `np.expm1` inverse transformation) was used during cross-validation to stabilize target distribution, but only the raw, unfitted `xgb_model` was serialized using `joblib.dump(xgb_model)`.  
**Failure Mode:** The serialized model predicted raw log-prices ($\approx 8.6$ to $9.2$) instead of real Indian Rupee values ($\approx ₹5,500$ to $₹10,000$). Predictions were in the wrong unit of measure by three orders of magnitude.  
**Why It Wasn't Caught:** Cross-validation metrics appeared high because CV utilized the wrapper, while the dumped artifact was never subjected to an integration test checking whether predictions fell within realistic INR bounds ($₹1,000$ to $₹100,000$).  
**Fix:** Enclosed the `ColumnTransformer` preprocessor and the `TransformedTargetRegressor(regressor=XGBRegressor(...))` inside a single unified `sklearn.pipeline.Pipeline`. Calling `joblib.dump(pipeline)` ensures preprocessing, tree ensembles, and target inverse transformations are atomically bundled.

### Bug #5 — Miscellaneous Reproducibility Issues
- **No random seeds:** Injected `RANDOM_SEED = 42` across `train_test_split`, `XGBRegressor`, and cross-validation generators.
- **Rare airline categories:** Mapped low-frequency airlines ("Jet Airways Business", "Trujet", "Multiple carriers premium economy") to `"Other"` prior to one-hot encoding, preventing dimension mismatch or unseen label errors at inference time.
- **No data validation:** Added runtime shape checks ($\ge 100$ rows) and column existence checks inside `load_raw_data()`.

---

## 2. Architecture Decisions

### 2.1 ONNX Export Strategy — Approach A vs Approach B

**Approach B (Rejected):** Attempting to export the complete Scikit-Learn Pipeline (including `ColumnTransformer` and `TransformedTargetRegressor`) using `skl2onnx`.  
**Problem:** `skl2onnx` does not natively support `TransformedTargetRegressor` or custom function transformers. Workarounds require monkey-patching internal converter registries, which introduces fragility and breaks across package upgrades.

**Approach A (Chosen):** Export the inner trained `XGBRegressor` to ONNX using `onnxmltools.convert_xgboost`, and serialize the fitted `ColumnTransformer` separately as `models/preprocessor.joblib`.  
**Inference Flow:**
1. `engineer_features(input_df)` $\rightarrow$ computes calendar, duration, route, and distance features.
2. `preprocessor.transform(X)` $\rightarrow$ performs one-hot encoding and standard scaling in Scikit-Learn.
3. ONNX Runtime inference $\rightarrow$ executes tree traversals on CPUExecutionProvider, returning log-scale predictions.
4. `np.expm1(log_price)` $\rightarrow$ maps back to Indian Rupee (INR) space.

**Trade-off:** Preprocessing requires Python and Scikit-Learn. If pure language-agnostic C++/Rust deployment is required, preprocessing steps must be converted into ONNX operators. However, for a containerized Python microservice, Approach A provides substantial inference speedups with zero conversion risk.

### 2.2 Abstract Base Class for Predictors

**Why ABC:** The FastAPI web layer interacts with the model purely through `predictor.predict(df)` and `predictor.predict_safe(df)`. Inheriting from `BaseFlightPredictor(ABC)` guarantees at instantiation time that both `SklearnPredictor` and `OnnxPredictor` adhere strictly to the expected interface. If a required method is omitted or mistyped, Python raises an explicit `TypeError` upon startup rather than a runtime `AttributeError` during a live production request.

**Alternative considered:** Duck typing without an ABC. Rejected because it lacks formal interface documentation and delays signature mismatches to invocation time.

---

## 3. Metrics

| Metric | Baseline (notebook) | After fixes |
|--------|:-------------------:|:-----------:|
| R² (INR space) | 0.7598 | **0.8449** |
| RMSE (INR) | Not recorded | **₹1,812** |
| MAE (INR) | Not recorded | **₹1,144** |
| Threshold met (≥ 0.80) | ❌ | **✅** |

**Why R² improved:** The target variable (ticket price) exhibits positive skew. By training XGBoost via `TransformedTargetRegressor(func=np.log1p)`, errors are penalized proportionally across price ranges rather than being dominated exclusively by expensive luxury flights. Combined with engineered temporal features and coordinate Euclidean distances, the model generalized effectively on held-out test data.

---

## 4. Verification Evidence

```bash
# 1. Training Pipeline Execution
$ flight-train
{"event": "Starting flight price model training", "level": "info", "logger": "flight_predictor.train"}
{"event": "Data loaded sucessfully", "level": "info", "logger": "flight_predictor.data"}
{"event": "Cross-validation complete", "level": "info", "logger": "flight_predictor.train", "cv_mean_r2": 0.8385}
{"event": "Training complete. Artifact saved.", "level": "info", "logger": "flight_predictor.train"}

 Training Complete!
 R2 = 0.8449
 RMSE = 1812
 MAE = 1144

# 2. Test Suite & Coverage
$ pytest tests/test_features.py tests/test_api.py -v --cov=flight_predictor --cov-fail-under=70
======================== 38 passed in 5.59s ========================
Required test coverage of 70% reached. Total coverage: 73.63%

# 3. Health Check
$ curl -s http://localhost:8888/health | python -m json.tool
{
    "status": "healthy",
    "model_loaded": true,
    "predictor_type": "SklearnPredictor",
    "version": "1.0.0"
}

# 4. Live REST API Prediction
$ curl -s -X POST http://localhost:8888/predict \
  -H "Content-Type: application/json" \
  -d '{
    "airline": "IndiGo",
    "date_of_journey": "15/06/2024",
    "source": "Delhi",
    "destination": "Bangalore",
    "dep_time": "06:00",
    "arrival_time": "08:45",
    "duration": "2h 45m",
    "total_stops": "non-stop",
    "additional_info": "No info"
  }' | python -m json.tool
{
    "predicted_price_inr": 4812.7,
    "predictor_type": "SklearnPredictor",
    "status": "success",
    "request_id": "94714eea-bcf4-4a08-a511-53f09c94b7a4"
}
```

---

## 5. What I Would Do Next

1. **Hyperparameter Optimization with Optuna:** Conduct Bayesian optimization over `colsample_bytree`, `learning_rate`, `max_depth`, and `min_child_weight` logged directly to MLflow.
2. **Feature Engineering Expansion:** Calculate "days until departure" (booking advance window) and holiday calendar flags to account for surge pricing dynamics.
3. **Data Drift & Monitoring:** Deploy Evidently AI to continuously track Population Stability Index (PSI) and Kolmogorov-Smirnov statistics on inference payloads.
