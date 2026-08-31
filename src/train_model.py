"""
train_model.py
==============
End-to-End QSPR Pipeline with Selective Feature Scaling and Grid Search.
Applies StandardScaler exclusively to continuous RDKit descriptors using
ColumnTransformer to avoid altering binary Morgan Fingerprints.
Evaluates Ridge, SVR, Random Forest, and XGBoost models using Repeated K-Fold CV.
"""

# ----------------------------------------------------------------------
# Imports & Configuration
# ----------------------------------------------------------------------
import os
import sys
import json
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor
import joblib

# Configure Terminal Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Parameters & Output Setup
# ----------------------------------------------------------------------
INPUT_CSV = "data/processed/featurized_dataset_curated.csv"
OUTPUT_DIR = "data/processed/model_results_reduced_curated"
RANDOM_STATE = 42
TEST_SIZE = 0.20

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Load & Prepare Dataset
# ----------------------------------------------------------------------
log.info("=" * 75)
log.info("ChemPyQSPR - Hyperparameter Grid Search & Model Evaluation")
log.info("=" * 75)

if not os.path.exists(INPUT_CSV):
    log.error("Input dataset '%s' not found. Aborting execution.", INPUT_CSV)
    sys.exit(1)

log.info("Loading feature dataset from: %s", INPUT_CSV)
df = pd.read_csv(INPUT_CSV)

target_col = 'flash_point'
if target_col not in df.columns:
    possible_targets = [c for c in df.columns if 'flash' in c.lower()]
    if possible_targets:
        target_col = possible_targets[0]
    else:
        log.error("Target column for Flash Point not found.")
        sys.exit(1)

ignore_cols = ['name', 'smiles', target_col, 'cid', 'cas', 'primary_functional_group', 'all_functional_groups']
feature_cols = [col for col in df.columns if col not in ignore_cols]

# Clean and extract numerical data
df_clean = df[feature_cols + [target_col]].apply(pd.to_numeric, errors='coerce').dropna()
X = df_clean[feature_cols]
y = df_clean[target_col].values

# ----------------------------------------------------------------------
# 2. Identify Feature Subsets for Selective Preprocessing
# ----------------------------------------------------------------------
# Separate continuous physical-chemical descriptors from binary Morgan fingerprints
continuous_cols = [c for c in feature_cols if not c.startswith("fp_")]
fingerprint_cols = [c for c in feature_cols if c.startswith("fp_")]

log.info("Dataset loaded successfully: N = %d valid compounds", len(df_clean))
log.info("Target Variable : '%s' (Min: %.2f deg C | Mean: %.2f deg C | Max: %.2f deg C)",
         target_col, y.min(), y.mean(), y.max())
log.info("Feature Breakdown : %d Continuous Descriptors | %d Morgan Fingerprint Bits",
         len(continuous_cols), len(fingerprint_cols))

# Evaluate VarianceThreshold filter stats for explicit logging
fp_selector_check = VarianceThreshold(threshold=0.01)
fp_selector_check.fit(X[fingerprint_cols])
retained_fp_count = int(fp_selector_check.get_support().sum())
discarded_fp_count = len(fingerprint_cols) - retained_fp_count
total_retained_features = len(continuous_cols) + retained_fp_count

log.info("Variance Filtering: Discarded %d low-variance fingerprint bits (< 1.0%% variance) | Retained %d total features (%d continuous + %d fingerprints)",
         discarded_fp_count, total_retained_features, len(continuous_cols), retained_fp_count)

# ColumnTransformer: Scale continuous descriptors, apply VarianceThreshold to binary fingerprints
preprocessor = ColumnTransformer(
    transformers=[
        ("num_scaler", StandardScaler(), continuous_cols),
        ("fp_selector", VarianceThreshold(threshold=0.01), fingerprint_cols)
    ],
    remainder="drop"
)

# ----------------------------------------------------------------------
# 3. Stratified Train / Test Split (80/20)
# ----------------------------------------------------------------------
# Stratify by Flash Point quartiles to ensure balanced distribution
bins = pd.qcut(y, q=4, labels=False, duplicates="drop")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=bins
)

log.info("Data split: %d samples for Training (80%%) | %d samples for Testing (20%%)", 
         len(y_train), len(y_test))

# ----------------------------------------------------------------------
# 4. Cross-Validation & Hyperparameter Grids Definition
# ----------------------------------------------------------------------
rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)

# Estimators & Hyperparameter Grids
models_grid_config = {
    "Ridge Regression": {
        "pipeline": Pipeline([
            ("preprocessor", preprocessor),
            ("model", Ridge(random_state=RANDOM_STATE))
        ]),
        "param_grid": {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]
        }
    },
    "Support Vector Regressor (SVR)": {
        "pipeline": Pipeline([
            ("preprocessor", preprocessor),
            ("model", SVR(kernel="rbf"))
        ]),
        "param_grid": {
            "model__C": [10.0, 50.0, 100.0],
            "model__epsilon": [0.1, 1.0, 2.0],
            "model__gamma": ["scale", "auto"]
        }
    },
    "Random Forest": {
        "pipeline": Pipeline([
            ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))
        ]),
        "param_grid": {
            "model__n_estimators": [50, 100, 150],
            "model__max_depth": [4, 5, 6],
            "model__min_samples_split": [2, 3],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": [0.6, 0.8, 1.0]
        }
    },
    "XGBoost": {
        "pipeline": Pipeline([
            ("model", XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0))
        ]),
        "param_grid": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 4, 5],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__reg_lambda": [1.0, 3.0, 5.0],
            "model__subsample": [0.7, 0.9]
        }
    }
}

# ----------------------------------------------------------------------
# 5. Execute GridSearchCV & Evaluate on Test Set
# ----------------------------------------------------------------------
log.info("-" * 75)
log.info("Starting GridSearchCV across models using Repeated 5-Fold (5 repeats)...")
log.info("-" * 75)

results = []
best_estimators = {}

for name, config in models_grid_config.items():
    log.info("Running Grid Search for: %s...", name)
    
    grid = GridSearchCV(
        estimator=config["pipeline"],
        param_grid=config["param_grid"],
        cv=rkf,
        scoring="r2",
        n_jobs=-1
    )
    
    # Fit Grid Search on Training Set
    grid.fit(X_train, y_train)
    
    best_cv_r2 = grid.best_score_
    best_params = grid.best_params_
    best_pipe = grid.best_estimator_
    best_estimators[name] = best_pipe
    
    log.info("  * Best CV R2   : %.4f", best_cv_r2)
    log.info("  * Best Params  : %s", best_params)
    
    # Evaluate best estimator on independent test set (20%)
    y_pred_test = best_pipe.predict(X_test)
    
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = root_mean_squared_error(y_test, y_pred_test)
    
    log.info("  * Test Results : R2=%.4f | MAE=%.2f deg C | RMSE=%.2f deg C",
             r2_test, mae_test, rmse_test)
    
    clean_params = {k.split("__")[1]: v for k, v in best_params.items()}
    
    results.append({
        "Model": name,
        "Best_CV_R2_Train": round(best_cv_r2, 4),
        "Test_R2": round(r2_test, 4),
        "Test_MAE (deg C)": round(mae_test, 2),
        "Test_RMSE (deg C)": round(rmse_test, 2),
        "Best_Parameters": json.dumps(clean_params)
    })

# ----------------------------------------------------------------------
# 6. Results Compilation & Export
# ----------------------------------------------------------------------
results_df = pd.DataFrame(results).sort_values(by="Test_R2", ascending=False).reset_index(drop=True)

# Export metrics to CSV and JSON
metrics_csv_path = os.path.join(OUTPUT_DIR, "model_metrics_gridsearch.csv")
metrics_json_path = os.path.join(OUTPUT_DIR, "model_metrics_gridsearch.json")

results_df.to_csv(metrics_csv_path, index=False)

with open(metrics_json_path, "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "n_samples": int(len(df_clean)),
        "n_continuous_features": int(len(continuous_cols)),
        "n_fingerprint_features": int(len(fingerprint_cols)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "cv_scheme": "RepeatedKFold (5 splits, 5 repeats)",
        "results": results_df.to_dict(orient="records")
    }, f, indent=2)

champion_model = results_df.iloc[0]

log.info("=" * 75)
log.info("GRID SEARCH & EVALUATION COMPLETED SUCCESSFULLY!")
log.info("Summary Table:\n\n%s\n", results_df[["Model", "Best_CV_R2_Train", "Test_R2", "Test_MAE (deg C)", "Test_RMSE (deg C)"]].to_string(index=False))
log.info("CHAMPION MODEL : %s", champion_model["Model"])
log.info("  * Best CV R2   : %.4f", champion_model["Best_CV_R2_Train"])
log.info("  * Test R2      : %.4f", champion_model["Test_R2"])
log.info("  * Test MAE     : %.2f deg C", champion_model["Test_MAE (deg C)"])
log.info("  * Test RMSE    : %.2f deg C", champion_model["Test_RMSE (deg C)"])
log.info("  * Parameters   : %s", champion_model["Best_Parameters"])
log.info("Metrics saved at  : %s", metrics_csv_path)

# ----------------------------------------------------------------------
# 7. Champion Model Artifacts Serialization (models/)
# ----------------------------------------------------------------------
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

champion_name = champion_model["Model"]
champion_pipeline = best_estimators[champion_name]

best_model_path   = os.path.join(MODELS_DIR, "best_model.pkl")
feature_cols_path = os.path.join(MODELS_DIR, "feature_cols.pkl")
scaler_path       = os.path.join(MODELS_DIR, "scaler.pkl")

# 1. Save Champion Pipeline (self-contained with preprocessor and model)
joblib.dump(champion_pipeline, best_model_path)
log.info("Saved champion model pipeline to: %s", best_model_path)

# 2. Save Feature Columns List (ordered feature names expected by inference)
joblib.dump(feature_cols, feature_cols_path)
log.info("Saved feature columns list to   : %s (%d features)", feature_cols_path, len(feature_cols))

# 3. Fit and Save Standalone StandardScaler (for fallback/bare estimators)
standalone_scaler = StandardScaler()
standalone_scaler.fit(X_train)
joblib.dump(standalone_scaler, scaler_path)
log.info("Saved fitted StandardScaler to  : %s", scaler_path)

# Also save copies into OUTPUT_DIR for experiment tracking
joblib.dump(champion_pipeline, os.path.join(OUTPUT_DIR, "best_model.pkl"))
joblib.dump(feature_cols, os.path.join(OUTPUT_DIR, "feature_cols.pkl"))
joblib.dump(standalone_scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

log.info("=" * 75)