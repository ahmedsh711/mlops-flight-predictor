"""
Training pipeline for the flight price prediction model.

Key design decisions:
1. TransformedTargetRegressor wraps XGBoost — handles log1p/expm1 automatically
2. Pipeline combines preprocessor + model — saved as one artifact
3. R² threshold enforced — prevents deploying a worse model
4. Seeded for reproducibility — same result every run
"""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor as TTR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from flight_predictor.config import (
    MODELS_DIR,
    PIPELINE_PATH,
    MODEL_INFO_PATH,
    PRICE_COLUMN,
    RANDOM_SEED,
    R2_THRESHOLD,
    TEST_SIZE,
    CV_FOLDS
)

from flight_predictor.data import load_raw_data
from flight_predictor.features import (
    engineer_features,
    build_preprocessor,
    get_feature_columns
)

logger = logging.getLogger(__name__)

def build_pipeline(xgb_params: dict[str, Any] | None = None) -> Pipeline:
    if xgb_params is None:
        xgb_params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "tree_method": "hist",   
        }

    preprocessor = build_preprocessor()

    # TTR wraps XGBoost — applies log1p before fit, expm1 after predict
    ttr_model = TTR(
        regressor = XGBRegressor(**xgb_params),
        func = np.log1p,
        inverse_func = np.expm1
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ('model', ttr_model)
    ])


def evaluate_model(pipeline : Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    return {
        "r2_inr_space": round(r2, 4),
        "rmse_inr": round(rmse, 4),
        "mae_inr": round(mae, 4)
    }

def train(data_path: Path | None = None) -> dict[str, Any]:
    """
    Full training pipeline: load → engineer → split → train → evaluate → save.

    Returns:
        Dictionary of metrics and artifact paths.

    Raises:
        ValueError: If trained model fails R² threshold.
    """

    #1- Load and validate raw data
    logger.info("Starting flight price model training")
    df_raw = load_raw_data(data_path)

    # 2- Feature_Engineering:
    df = engineer_features(df_raw)
    feature_cols = get_feature_columns()

    # validate all feature columns exist after engineering
    missing_feats = set(feature_cols) - set(df.columns)
    if missing_feats:
        raise ValueError(f"Missing feature columns after engineering: {missing_feats}")

    X = df[feature_cols]
    y = df[PRICE_COLUMN].astype(float)

    # 3- Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = TEST_SIZE, random_state=RANDOM_SEED
    )

    logger.info(
        "Data Split Complete",
        extra = {"train_size": len(X_train), "test_size": len(X_test)}
    )

    # 4- Build pipeline
    pipeline = build_pipeline()

    # 5- Cross_validation
    logger.info(f"Running {CV_FOLDS}-fold cross-validation..")
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=CV_FOLDS, scoring="r2", n_jobs = -1
    )
    logger.info(
        "Cross-validation complete",
        extra={
            "cv_mean_r2": round(cv_scores.mean(), 4),
            "cv_std_r2" : round(cv_scores.std(), 4)
        }
    )

    # 6- Final training on full training set:
    pipeline.fit(X_train, y_train)

    # 7- Evaluate on held-out test set
    metrics = evaluate_model(pipeline, X_test, y_test)
    logger.info("Test evaluation Complete", extra=metrics)

    # 8- Enforce Quality Gate:
    r2 = metrics["r2_inr_space"]
    if r2 < R2_THRESHOLD:
        raise ValueError(
            f"Model R2 = {r2:.4f} is below threshold {R2_THRESHOLD}. "
            "Review your features and hyperparameters before deploying."
        )

    # 9- Saving Artifact:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)

    # Save preprocessor separetly (for onnx export):
    preprocessor_fitted = pipeline.named_steps["preprocessor"]
    joblib.dump(preprocessor_fitted, MODELS_DIR/ "preprocessor.joblib")

    # save model info
    model_info = {
        "metrics": metrics,
        "cv_mean_r2": round(cv_scores.mean(), 4),
        "cv_std_r2": round(cv_scores.std(), 4),
        "feature_count": X_train.shape[1],
        "train_size": len(X_train),
        "r2_threshold": R2_THRESHOLD,
        "random_seed": RANDOM_SEED,
        "xgb_params": {
            k: (None if isinstance(v, float) and np.isnan(v) else v)
            for k, v in pipeline.named_steps["model"].regressor.get_params().items()
        }
    }

    with open(MODEL_INFO_PATH, "w") as f:
        json.dump(model_info, f, indent=2)

    logger.info(
        "Training complete. Artifact saved.",
        extra={"pipeline_path": str(PIPELINE_PATH), **metrics}
    )

    return model_info

def main() -> None:
    from flight_predictor.logging_conf import setup_logging
    setup_logging()
    info = train()
    print(f"\n Training Complete!")
    print(f" R2 = {info['metrics']['r2_inr_space']:.4f}")
    print(f" RMSE = {info['metrics']['rmse_inr']:.0f}")
    print(f" MAE = {info['metrics']['mae_inr']:.0f}")

if __name__ == '__main__':
    main()