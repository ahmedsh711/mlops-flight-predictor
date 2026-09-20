# Module 2 Report — Experiment Tracking, Data Versioning & CI/CD
**Student:** Ahmed Al-Shobaki  
**Date:** September 20, 2026  
**Builds on:** Module 1 (packaged project, FastAPI, Docker)

---

## Executive Summary

Building on the Module 1 foundation, I implemented an automated MLOps Level 2 infrastructure across three core pillars:
1. **MLflow Experiment Tracking & Model Registry**: Every training run, hyperparameter set, evaluation metric, and model version is recorded and tracked against a centralized server (hosted on DagsHub with local fallback).
2. **Data Versioning with DVC & AWS S3**: Raw and processed datasets are decoupled from Git history, pinned to deterministic content hashes (`ea995518`), and stored remotely on AWS S3 (`s3://my-flight-predictor-dvc-bucket/dvc-storage`).
3. **Automated CI/CD & Continuous Training (CT)**: A multi-stage GitHub Actions pipeline validates code syntax, enforces ≥70% unit test coverage, verifies model quality gates, builds production Docker images to GHCR, and supports automated scheduled/drift-triggered retraining with human approval before Staging promotion.

---

## 1. Experiment Tracking (MLflow)

### What was the problem before MLflow?

Prior to integrating MLflow, model development suffered from complete loss of experiment history:
- Each execution of the notebook or training script silently overwritten `models/pipeline.joblib`.
- Comparing whether a learning rate of 0.01 outperformed 0.05 required manual logging in text files or terminal scrollback.
- It was impossible to trace which commit, dataset version, or hyperparameter combination produced an artifact currently in production.
- Artifacts lacked lifecycle management: there was no staging-versus-production designation or rollback mechanism.

### What does MLflow record for each run?

For each of the three configurations evaluated in `experiments/train_with_mlflow.py`:
- **Hyperparameters:** `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `random_seed`, `test_size`, `cv_folds`, `n_features`, `n_train`, `n_test`.
- **Metrics:** `r2_inr_space`, `rmse_inr`, `mae_inr`, `cv_mean_r2`, `cv_std_r2`.
- **Artifacts:** Serialized Scikit-learn Pipeline (`flight_price_model/`), feature importance plot (`feature_importance.png`), and environment metadata.
- **Tags & Metadata:** `config_name`, `model_type`, `target_transform`, `description`.

### Why log metrics in INR space, not log space?

While the model internally minimizes root mean squared error on $\ln(\text{price} + 1)$, reporting log-space errors (e.g., $\text{MAE} = 0.14$) fails business interpretability:
- A non-technical stakeholder cannot evaluate the financial risk of a $0.14$ log error.
- Because the exponential transform is non-linear, a uniform log error translates to vastly different rupee deviations depending on ticket price.
- Logging metrics in Indian Rupee space ($\text{MAE} = ₹1,144$, $\text{RMSE} = ₹1,812$) provides direct, actionable business context: on average, a prediction deviates by ₹1,144 from the actual fare.

### Best run results

| Config | Parameters | R² (INR space) | RMSE (INR) | MAE (INR) |
|--------|------------|:--------------:|:----------:|:---------:|
| **`xgb_baseline`** | $N=300, \eta=0.05, \text{depth}=6, \text{sub}=0.8$ | **0.8449** | **₹1,811.82** | **₹1,144.03** |
| **`xgb_aggressive`** | $N=200, \eta=0.01, \text{depth}=8, \text{sub}=0.9$ | 0.7604 | ₹2,252.06 | ₹1,400.75 |
| **`xgb_regularised`** | $N=500, \eta=0.03, \text{depth}=5, \alpha=0.1, \lambda=1.5$ | 0.8339 | ₹1,874.71 | ₹1,211.75 |

**Best config:** `xgb_baseline` ($R^2 = 0.8449$, $\text{RMSE} = ₹1,811.82$)  
**Promoted to Staging:** **Yes**. Because $R^2 = 0.8449 \ge 0.80$, the winning run (`7c99d27a462c456387f13d41bf5c166f`) was registered under the model name `FlightPricePredictor` and promoted to `Staging` stage in the MLflow Model Registry.

---

## 2. Data Versioning (DVC)

### DVC tag reference

| DVC/Git tag | Description | Hash (first 8 chars) | Remote Target |
|-------------|-------------|:--------------------:|:-------------:|
| `v1.0-original` | Original `flight_data.csv` raw dataset (10,683 rows) | `ea995518` | `s3://my-flight-predictor-dvc-bucket/dvc-storage` |

### How DVC connects to MLflow

By tracking `data/raw/flight_data.csv.dvc` in Git alongside the training pipeline code, DVC and MLflow form a closed-loop provenance graph:
1. **Git Commit** captures exact source code and the `.dvc` hash pointer.
2. **DVC Remote (AWS S3)** stores immutable data blocks indexed by content hash.
3. **MLflow Run** logs training parameters, model weights, and Git commit SHA.
4. **Reproducibility Guarantee:** Any historical model version can be perfectly reproduced by checking out the corresponding Git commit and executing `dvc pull`.

### What would break without DVC?

Without DVC:
- Large dataset files (>100 MB) would either bloat Git history beyond GitHub's limits or be excluded into `.gitignore` without any version control.
- If training data is updated or modified locally, team members and cloud CI runners have no reliable mechanism to pull identical data.
- Debugging model drift or performance regressions across historical models becomes impossible when data snapshots are lost.

---

## 3. CI/CD Pipeline

### Pipeline flow

```
Push to main / Pull Request
        │
        ▼
[Job 1: Lint & Format Check] — ruff check + ruff format --check
        │ (halts pipeline if formatting or lint rules fail)
        ▼
[Job 2: Unit & Integration Tests] — pytest (38 tests)
        │ --cov=flight_predictor --cov-fail-under=70
        │ (halts pipeline if any test fails or coverage < 70%)
        ▼
[Job 3: Model Quality Gate] — dvc pull data → flight-train → quality_gate.py
  (main branch only)          Validates R² ≥ 0.80 & regression ≤ 0.02
        │ (halts pipeline before Docker packaging if model degrades)
        ▼
[Job 4: Docker Build & Push] — multi-stage container build
  (main branch only)           Pushes ghcr.io/ahmedsh711/flight-predictor:latest
```

### Quality gate thresholds

| Check | Threshold | Rationale |
|-------|:---------:|-----------|
| **R² Floor** | $\ge 0.80$ | Absolute minimum business requirement defined in project specification. |
| **Max Regression** | $\le 0.02$ | Prevents deploying a newly trained model if it is >2% worse than the previous best model. |
| **Test Coverage** | $\ge 70\%$ | Ensures feature engineering, data loading, and inference wrappers are verified (current: **73.63%**). |

### What does the CI pipeline prevent?

1. **Syntax and Code Quality Regressions:** Enforces PEP8, import sorting, and linting cleanliness across `src/`, `api/`, `tests/`, `experiments/`, and `scripts/`.
2. **Silent Duration Parser Breaks:** Guarantees that short flights ("45m", "5m") never re-introduce index errors.
3. **Model Metric Regressions:** Prevents any automated push or merge from publishing a model if $R^2 < 0.80$ or if regression exceeds 2%.
4. **Broken Docker Container Images:** Verifies that container builds compile and pass dependencies before reaching the GitHub Container Registry.

---

## 4. Continuous Training

### Triggers configured in `continuous-training.yml`

1. **Schedule (`cron: "0 2 * * 1"`):** Automatically initiates retraining every Monday at 02:00 UTC (07:30 IST) to refresh pricing patterns following weekend travel demand.
2. **Manual (`workflow_dispatch`):** Enables authorized engineers to manually trigger retraining from the GitHub UI with a required audit reason.
3. **Repository Dispatch (`drift-detected`):** Listens for external automated webhook events dispatched when production monitoring services detect feature or prediction drift.

### Human approval gate

Rather than auto-deploying retrained models directly to production, the pipeline utilizes a GitHub Environment protection rule:
- **Environment:** `staging`
- **Required Reviewer:** `ahmedsh711`
- **Mechanism:** When Job 1 (`retrain`) finishes successfully, Job 2 (`approve-staging`) pauses execution. The model version remains in stage `"None"`. Only after human review and approval in GitHub Actions does Job 3 (`promote-staging`) execute, transitioning the model to `"Staging"` in MLflow.

### Session 5 Drift Detection Blueprint

In Session 5, the automated drift architecture will operate as follows:
1. Production FastAPI instances stream inference payloads and predictions to a persistent log/database.
2. An Evidently AI background worker runs scheduled distribution comparisons using `DataDriftPreset`.
3. If Population Stability Index (PSI) exceeds $0.20$ or two-sample Kolmogorov-Smirnov p-values drop below $0.05$ on key drivers (e.g., duration, airline distribution), the service issues an authenticated POST request to GitHub's `/repos/{owner}/{repo}/dispatches` endpoint with `event_type: "drift-detected"`.
4. GitHub Actions triggers `continuous-training.yml` to ingest newly labeled flight records from S3 and retrain the ensemble.

---

## 5. Reflection

### What surprised you most in Session 2?

The biggest revelation was the fragility of unpinned upstream dependencies and the exact nature of data-code pairing. For instance, an unpinned `pygtrie` update broke internal trie representations in `sqltrie`/`dvc`, emphasizing why lockfiles (`uv.lock`) and explicit constraints are essential in production environments. Additionally, realizing that CI/CD in MLOps must validate **data and model quality**, not just code syntax, transformed how I view deployment pipelines.

### If you had one more week, what would you add?

1. **Automated Shadow Deployment:** Deploy newly promoted staging models as shadow models behind the production API, evaluating live traffic metrics in parallel before traffic cutover.
2. **Bayesian Hyperparameter Sweeps:** Integrate Optuna sweeps directly into the weekly retraining pipeline to adapt tree depths and regularization parameters to shifting datasets.
3. **Automated Rollback Webhooks:** Configure Prometheus alerting to automatically transition the MLflow registry back to the previous stable production version if API inference error rates exceed 1%.
