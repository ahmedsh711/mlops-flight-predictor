# ✈️ End-to-End MLOps Flight Price Predictor

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data_Versioning-blue.svg)](https://dvc.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![CI/CD](https://github.com/ahmedsh711/mlops-flight-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmedsh711/mlops-flight-predictor/actions)

A complete, production-ready Machine Learning Operations (MLOps) pipeline for predicting domestic flight prices in India. This project demonstrates advanced data science feature engineering coupled with enterprise-grade MLOps automation, continuous training, and dynamic model serving.

## 🌟 Key Features

*   **Robust ML Pipeline:** Custom `ColumnTransformer` (OneHotEncoding, StandardScaler) unified with an `XGBoost` regressor to prevent training-serving skew.
*   **Target Transformation:** Wraps the model in a `TransformedTargetRegressor` (using `log1p`) to handle heavily right-skewed flight prices, drastically improving $R^2$.
*   **Spatial & Temporal Engineering:** Hand-crafted Haversine distance calculations and advanced temporal extractions from raw timestamps.
*   **Data & Model Versioning (DVC):** Tracks raw data, processed data, and serialized models (`.joblib`, `.onnx`) using DVC backed by AWS S3.
*   **Experiment Tracking (MLflow):** Integrates with DagsHub MLflow to automatically log hyperparameters, metrics, and models.
*   **Continuous Training (CT):** A fully automated GitHub Actions workflow that pulls data, repros the DVC pipeline, trains the model, exports it to ONNX, and pushes the new lockfile back to Git.
*   **Dynamic Model Registry:** CI/CD automatically promotes models to the `Staging` stage if they pass the $R^2 \ge 0.80$ quality gate.
*   **Lightning Fast Deployment:** Uses `uv` for 17-second dependency resolution on Streamlit Cloud, dynamically fetching the latest model from the MLflow registry on boot.

## 🏗️ System Architecture

```mermaid
graph TD
    classDef data fill:#f4f4f4,stroke:#333,stroke-width:2px;
    classDef process fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef cicd fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef registry fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef deploy fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;

    subgraph Data Layer
        A[(Raw Data)]:::data
        B[(AWS S3 / DVC Remote)]:::data
        A -->|Tracked by| B
    end

    subgraph GitHub Actions CI/CD Pipeline
        C{Push / Manual Trigger}:::cicd
        D[Lint & Test Ruff/Pytest]:::cicd
        E[DVC Pull Latest Data]:::cicd
        F[DVC Repro Preprocess]:::process
        
        C --> D --> E --> F
        E -->|Downloads Data| A
    end

    subgraph Training Pipeline
        G[Feature Engineering]:::process
        H[XGBoost Training pipeline.joblib]:::process
        I[Quality Gate R2 >= 0.80]:::process
        
        F --> G --> H --> I
        H -->|DVC Push Artifacts| B
    end

    subgraph MLflow on DagsHub
        J[(Experiment Tracking)]:::registry
        K[(Model Registry)]:::registry
        L([Promote to Staging]):::registry
        
        I -->|Log Metrics| J
        I -->|If Pass| K --> L
    end

    subgraph Production Deployment
        M[FastAPI Backend]:::deploy
        N[Streamlit Cloud UI]:::deploy
        
        L -.->|Dynamically Fetches Model| M
        L -.->|Dynamically Fetches Model| N
        M <-->|API Requests| N
    end
```

## 📂 Repository Structure

```text
.
├── .github/workflows/       # CI/CD and Continuous Training Actions
├── data/
│   ├── raw/                 # Raw CSV files (tracked by DVC)
│   └── processed/           # Engineered features (tracked by DVC)
├── experiments/             # MLflow training scripts
├── models/                  # Serialized pipelines (joblib/ONNX)
├── src/flight_predictor/    # Core ML logic (features, train, evaluate)
├── api/                     # FastAPI backend
├── streamlit_app.py         # Streamlit UI
├── dvc.yaml                 # DVC pipeline definition
├── pyproject.toml           # Project metadata
└── uv.lock                  # Deterministic lockfile
```

## 🚀 Getting Started

### Prerequisites
*   Python 3.11+
*   [`uv`](https://github.com/astral-sh/uv) (Extremely fast Python package installer)
*   A DagsHub account (for MLflow remote tracking)

### 1. Installation
Clone the repository and install dependencies using `uv`:
```bash
git clone https://github.com/ahmedsh711/mlops-flight-predictor.git
cd mlops-flight-predictor
uv sync
source .venv/bin/activate
```

### 2. Pull Data and Models
The data and models are versioned with DVC. To download the latest versions from AWS S3:
```bash
dvc pull
```

### 3. Environment Variables
Create a `.env` file in the root directory for MLflow authentication:
```env
MLFLOW_TRACKING_URI="https://dagshub.com/ahmedsh711/mlops-flight-predictor.mlflow"
MLFLOW_TRACKING_USERNAME="your_dagshub_username"
MLFLOW_TRACKING_PASSWORD="your_dagshub_password_or_token"
```

### 4. Run the Streamlit Application
Start the interactive UI locally:
```bash
streamlit run streamlit_app.py
```
*Note: The app will automatically connect to DagsHub, download the latest `Staging` model version, and cache it in RAM for instant inference.*

### 5. Run the Training Pipeline
To retrain the model locally and track the experiment in MLflow:
```bash
dvc repro process
python experiments/train_with_mlflow.py
```

## 🛠️ Continuous Training (CI/CD)

This project features a fully automated Continuous Training (CT) loop. When triggered via GitHub Actions:
1. The runner installs dependencies and pulls raw data.
2. The ML pipeline is executed.
3. The model is evaluated. If $R^2 \ge 0.80$, it is registered to MLflow.
4. The model is exported to ONNX format.
5. New model artifacts are committed via DVC and pushed to AWS S3.
6. The updated `dvc.lock` is automatically committed and pushed back to the `main` branch.
7. The model is transitioned to `Staging`, where the Streamlit app seamlessly picks it up.
