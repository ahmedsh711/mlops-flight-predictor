# ✈️ Flight Price Predictor — MLOps Project Documentation

## 1. Project Overview

The **Flight Price Predictor** is an end-to-end Machine Learning Operations (MLOps) project designed to predict domestic flight prices in India based on historical booking data. The project emphasizes production-readiness, robust CI/CD, experiment tracking, and model lifecycle management.

### Key Capabilities
- **Automated Data Pipelines:** Built-in data extraction, preprocessing, and feature engineering.
- **Continuous Training (CT):** A fully automated GitHub Actions workflow to retrain the model when data or code changes.
- **Continuous Integration (CI):** Automated linting, testing, and coverage reporting.
- **Model Registry & Tracking:** Centralized experiment tracking using MLflow hosted on DagsHub.
- **Data Versioning:** S3-backed data and model versioning using DVC.
- **Interactive UI:** A sleek, user-friendly Streamlit application dynamically fetching the latest production-ready model.
- **API Serving:** A FastAPI backend serving real-time predictions.

---

## 2. Architecture & Tech Stack

### Frameworks & Tools
| Component | Tool/Framework | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.11 | Core programming language |
| **Package Manager** | `uv` | Ultra-fast dependency resolution and environment management |
| **Data Processing** | `pandas`, `numpy` | Data manipulation and numerical operations |
| **Machine Learning** | `scikit-learn`, `xgboost` | Pipeline construction, preprocessing, and model training |
| **Data Versioning** | `dvc` (with `s3` remote) | Tracking large datasets and serialized models |
| **Experiment Tracking** | `mlflow` | Logging metrics, parameters, and model artifacts |
| **Model Registry** | DagsHub MLflow | Centralized remote model storage and stage management |
| **API Backend** | `FastAPI`, `uvicorn` | High-performance REST API for model inference |
| **Frontend UI** | `Streamlit` | Interactive web interface deployed on Streamlit Cloud |
| **Testing** | `pytest`, `pytest-cov` | Unit and integration testing |
| **Linting/Formatting**| `ruff` | Extremely fast code linting and formatting |
| **CI/CD** | GitHub Actions | Automated testing and continuous training pipelines |

---

## 3. Design Decisions

### 1. `uv` for Dependency Management
We chose `uv` over `pip` or `poetry` for its massive speed advantages. By committing `uv.lock`, we ensure deterministic builds across local environments, CI runners, and deployment servers. 

### 2. DVC for Data, MLflow for Models
While DVC can version models, we adopted a hybrid approach:
- **DVC** tracks large binary files: `data/raw`, `data/processed`, and the final serialized pipelines (`pipeline.joblib`, `xgb_model.onnx`).
- **MLflow** handles the *logical* model lifecycle (e.g., transitioning models from `None` -> `Staging` -> `Production`) and experiment comparison.

### 3. Sklearn Pipeline Architecture
We encapsulated all preprocessing (OneHotEncoding, RobustScaling) and the estimator (XGBoost) into a single `sklearn.pipeline.Pipeline`. This prevents training-serving skew, as the exact same transformations applied during training are inherently applied during inference.

### 4. Dynamic Model Fetching in Production
The Streamlit app does **not** hardcode a model file. Instead, upon startup, it authenticates with DagsHub MLflow and dynamically downloads the latest model version tagged with the `Staging` alias. This decouples the application deployment from the model deployment.

---

## 4. CI/CD Workflows

The project features two primary GitHub Actions workflows:

### Continuous Integration (`ci.yml`)
Runs on every push and pull request.
1. Checks out the code.
2. Installs dependencies using `uv`.
3. Pulls required test data and models using `dvc pull --allow-missing`.
4. Runs `pytest` enforcing a strict minimum coverage threshold (40%).
5. Uploads coverage reports to Codecov.

### Continuous Training (`continuous-training.yml`)
Triggered manually (or via schedule/data updates).
1. Pulls the latest raw data via DVC.
2. Runs the data processing pipeline (`dvc repro process`).
3. Executes the MLflow training script (`train_with_mlflow.py`).
4. Evaluates the model against a Quality Gate (e.g., $R^2 \ge 0.80$).
5. If passed, registers the model in MLflow and exports it to ONNX (`flight-export`).
6. Pushes the new model artifacts to DVC (`dvc commit -f`, `dvc push`).
7. Commits the updated `dvc.lock` back to the repository.
8. Automatically transitions the newly registered MLflow model to the **Staging** environment.

---

## 5. Troubleshooting & Bug Fix History

Throughout development, we encountered and resolved several complex infrastructure issues:

### 1. DVC Overlap Errors in CI
**Issue:** The continuous training pipeline failed with `ERROR: cannot update 'pipeline.joblib': overlaps with an output of stage`.
**Fix:** Removed the manual `dvc add` command in the CI script. Since the models are outputs of `dvc.yaml` stages, we switched to using `dvc commit -f` to cleanly update the `dvc.lock` file without pipeline conflicts.

### 2. Missing DVC Dependencies in CI
**Issue:** `dvc commit` failed because intermediate files like `data/processed/features.parquet` or `models/xgb_model.onnx` did not exist on the ephemeral GitHub Actions runner.
**Fix:** Modified the CI pipeline to execute `dvc repro process` to generate missing intermediate data files locally on the runner, and added `flight-export` to ensure the ONNX model was generated before committing.

### 3. MLflow Sklearn Dependency Conflicts
**Issue:** `mlflow.sklearn.log_model()` failed during the pip environment inference step, throwing a `ResolutionImpossible` error for `scikit-learn` versions.
**Fix:** Disabled MLflow's automatic pip dry-run inference by explicitly passing `pip_requirements` into `log_model()`, dynamically fetching the exact installed versions using `importlib.metadata`.

### 4. Streamlit Cloud Model Loading Failures
**Issue:** The Streamlit app crashed on boot because it couldn't find the `pipeline.joblib` artifact.
**Fix:** 
1. Re-architected the app to pull the model securely from the remote DagsHub MLflow Registry using `mlflow.sklearn.load_model()`.
2. Updated MLflow API calls from the deprecated `get_latest_versions` to `search_model_versions` for MLflow 3.x compatibility.
3. Successfully promoted the fixed model to the `Staging` stage in MLflow so the Streamlit app could successfully locate it.

### 5. Streamlit Cloud Dependency Management
**Issue:** Streamlit Cloud failed to resolve dependencies because it prioritized `uv.lock` but missing core dependencies in `pyproject.toml`.
**Fix:** Injected `streamlit` and `mlflow` into the core `[dependencies]` array of `pyproject.toml` and regenerated `uv.lock`, ensuring seamless compatibility with Streamlit's `uv-sync` boot sequence.

---

## 6. Future Enhancements
- **Data Drift Detection:** Integrate evidently.ai to monitor incoming inference requests for data drift.
- **Production Promotion:** Add a manual approval workflow to transition models from `Staging` to `Production` in MLflow.
- **Containerization:** Write a `Dockerfile` for the FastAPI backend to allow deployment on Kubernetes or AWS ECS.
