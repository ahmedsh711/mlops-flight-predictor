import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from flight_predictor.config import (
    MODELS_DIR, PIPELINE_PATH, MODEL_INFO_PATH, 
    PRICE_COLUMN, RANDOM_SEED, R2_THRESHOLD, TEST_SIZE, CV_FOLDS
)
from flight_predictor.data import load_raw_data
from flight_predictor.features import (
    engineer_features, build_preprocessor, get_feature_columns
)
from flight_predictor.logging_conf import setup_logging
from flight_predictor.train import build_pipeline, evaluate_model

logger = logging.getLogger(__name__)

# Mlflow Configuration:
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"  )
EXPERIMENT_NAME = "flight-price-predictor"
REGISTERED_MODEL = "FlightPricePredictor"

# Hyperparameter configurations to try:
CONFIGS = [
    {
        "name": "xgb_baseline",
        "description": "Conservative : lower LR, more estimators",
        "params": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": RANDOM_SEED
        }
    },
    {
        "name": "xgb_aggressive",
        "description": "Higher LR, deeper trees ; may overfit!",
        "params": {
            "n_estimators": 200,
            "learning_rate": 0.01,
            "max_depth": 8,
            "subsample": 0.9,
            "min_child_weight": 3,
            "random_state": RANDOM_SEED
        }
    },
    {
        "name": "xgb_regularised",
        "description": "Strong regularization ; better generalization",
        "params": {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 5,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 1.5,
            "random_state": RANDOM_SEED
        }
    }
]

def train_with_tracking(
    config: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> tuple[str, float]:
    with mlflow.start_run(run_name=config["name"]) as run:
        # Log Configuration Metadata:
        mlflow.set_tag("config_name", config["name"])
        mlflow.set_tag("description", config["description"])
        mlflow.set_tag("model_type", "XGBoost + TransformedTargetRegressor")
        mlflow.set_tag("target_transform", "log1p/expm1")

        # Log hyperparameters
        mlflow.log_params(config['params'])
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])

        # Build pipeline with TTR
        pipeline = build_pipeline(config['params'])        

        # Cross validation
        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=CV_FOLDS, scoring='r2', n_jobs=-1
        )
        mlflow.log_metric("cv_mean_r2", cv_scores.mean())
        mlflow.log_metric("cv_std_r2", cv_scores.std())

        # Final fit
        pipeline.fit(X_train, y_train)

        # Evaluate in INR Space:
        metrics = evaluate_model(pipeline, X_test, y_test)
        mlflow.log_metric("r2_inr_space", metrics['r2_inr_space'])
        mlflow.log_metric("rmse_inr", metrics['rmse_inr'])
        mlflow.log_metric("mae_inr", metrics["mae_inr"])

        # Feature importance plot:
        try:
            import matplotlib.pyplot as plt
            xgb_model = pipeline.named_steps['model'].regressor_
            preprocessor = pipeline.named_steps['preprocessor']
            feature_names = (
                preprocessor.get_feature_names_out()
                if hasattr(preprocessor, "get_feature_names_out")
                else [f"feature_{i}" for i in range(len(xgb_model.feature_importances_))]
            )
            top_n = 20
            importances = xgb_model.feature_importances_
            top_indices = importances.argsort()[-top_n:][::-1]

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(
                [feature_names[i] for i in top_indices],
                importances[top_indices]
            )
            ax.set_title(f"Top {top_n} Features - {config['name']}")
            ax.set_xlabel("Importance")
            plt.tight_layout()
            mlflow.log_figure(fig, "feature_importance.png")
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Feature importance plot failed: {e}")

        # Log Sklearn Model:
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path='flight_price_model',
            serialization_format="cloudpickle",
            registered_model_name=REGISTERED_MODEL if metrics['r2_inr_space'] >= R2_THRESHOLD else None
        )

        logger.info(
            f"Run {config['name']} complete",
            extra={
                "run_id": run.info.run_id,
                **metrics
            }

        )
        return run.info.run_id, metrics['r2_inr_space']

def promote_best_model(best_run_id: str) -> None:
    client = mlflow.MlflowClient()

    # find the model version associated with this run
    versions = client.search_model_versions(f"run_id='{best_run_id}'")
    if not versions:
        logger.warning(f"No model version found for run {best_run_id}")
        return

    version = versions[0].version
    client.transition_model_version_stage(
        name=REGISTERED_MODEL,
        version=version,
        stage="Staging",
        archive_existing_versions=True
    )
    logger.info(
        f"Model version {version} promoted to staging",
        extra={"run_id": best_run_id, "model": REGISTERED_MODEL}
    )

def main() -> None:
    setup_logging()

    # Connect to Mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"Mlflow experiment: {EXPERIMENT_NAME} (ID: {experiment.experiment_id})")

    # Prepare data (once, shared across all runs for fair comparsion)
    df_raw = load_raw_data()
    df = engineer_features(df_raw)
    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df[PRICE_COLUMN].astype(float)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    # Run all configurations
    results: list[tuple[str, float]] = []
    for config in CONFIGS:
        logger.info(f"starting run: {config['name']}")
        run_id, r2 = train_with_tracking(config, X_train, X_test, y_train, y_test)
        results.append((run_id, r2))

    # Find and promote the best model
    best_run_id, best_r2 = max(results, key=lambda x: x[1])
    print(f"\nBest run: {best_run_id}")
    print(f"R2 = {best_r2:.4f}")

    if best_r2 >= R2_THRESHOLD:
        promote_best_model(best_run_id)
        print(f"Model promoted to staging in '{REGISTERED_MODEL}' registry")
    else:
        print(f"Best R2 ({best_r2:.4f}) below threshold ({R2_THRESHOLD}). Not promoting.")
    
if __name__ == '__main__':
    main()