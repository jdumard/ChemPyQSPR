"""
feature_importance.py
=====================
Extracts and visualizes Feature Importance using SHAP (SHapley Additive exPlanations)
and tree-based MDI (Mean Decrease in Impurity) for the ChemPyQSPR project.

This script relies on the XGBoost model (one of the top performers in grid search)
trained on the reduced feature space (VarianceThreshold > 0.01).

Outputs:
    Figures are saved to: data/processed/feature_importance/
    Top feature lists are exported to CSV.

Usage:
    Run from the project root:
        python src/feature_importance.py
"""

import os
import sys
import logging
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

INPUT_CSV = "data/processed/featurized_dataset.csv"
OUTPUT_DIR = "data/processed/feature_importance"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL = "flash_point"
RANDOM_STATE = 42

def run_shap_analysis():
    log.info("=" * 75)
    log.info("  ChemPyQSPR -- Feature Importance & SHAP Analysis")
    log.info("=" * 75)

    if not os.path.exists(INPUT_CSV):
        log.error("Input file not found: %s", INPUT_CSV)
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    
    ignore_cols = ['name', 'smiles', TARGET_COL, 'cid', 'cas', 'primary_functional_group', 'all_functional_groups']
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    
    df_clean = df[feature_cols + [TARGET_COL]].apply(pd.to_numeric, errors='coerce').dropna()
    X = df_clean[feature_cols]
    y = df_clean[TARGET_COL].values
    
    continuous_cols = [c for c in feature_cols if not c.startswith("fp_")]
    fingerprint_cols = [c for c in feature_cols if c.startswith("fp_")]
    
    log.info("Dataset loaded: N = %d compounds", len(df_clean))
    
    # Preprocessing identical to train_model_reduced.py
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), continuous_cols),
            ("fp", VarianceThreshold(threshold=0.01), fingerprint_cols)
        ],
        remainder="drop"
    )
    
    log.info("Applying Preprocessor (Standardization + VarianceThreshold)...")
    X_transformed = preprocessor.fit_transform(X)
    
    # Extract feature names after VarianceThreshold
    try:
        feature_names = preprocessor.get_feature_names_out()
        # Clean names (remove 'num__' and 'fp__' prefixes)
        feature_names = [name.split('__')[-1] for name in feature_names]
    except Exception as e:
        log.warning("Could not extract feature names directly: %s. Rebuilding manually.", e)
        selector = preprocessor.named_transformers_["fp"]
        mask = selector.get_support()
        selected_fp = np.array(fingerprint_cols)[mask].tolist()
        feature_names = continuous_cols + selected_fp
        
    log.info("Features after reduction: %d", len(feature_names))
    
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=RANDOM_STATE)
    
    # Train XGBoost with best grid search params
    xgb_params = {
        'learning_rate': 0.1,
        'max_depth': 5,
        'n_estimators': 300,
        'reg_lambda': 5.0,
        'subsample': 0.7,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    log.info("Training XGBoost Regressor for SHAP analysis...")
    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)
    
    # ---------------------------------------------------------
    # 1. XGBoost Native Feature Importance (Gain)
    # ---------------------------------------------------------
    log.info("Extracting native XGBoost Feature Importance (Gain)...")
    importances = model.feature_importances_
    xgb_imp_df = pd.DataFrame({"Feature": feature_names, "Importance_Gain": importances})
    xgb_imp_df = xgb_imp_df.sort_values(by="Importance_Gain", ascending=False).reset_index(drop=True)
    
    csv_path = os.path.join(OUTPUT_DIR, "01_xgboost_native_importance.csv")
    xgb_imp_df.to_csv(csv_path, index=False)
    log.info("Saved XGBoost native importance: %s", csv_path)
    
    # ---------------------------------------------------------
    # 2. SHAP Values Extraction
    # ---------------------------------------------------------
    log.info("Calculating SHAP values using TreeExplainer (this may take a moment)...")
    explainer = shap.TreeExplainer(model)
    # Use a background dataset to speed up plotting and analysis if necessary, but TreeExplainer is fast enough for 8000 rows.
    shap_values = explainer.shap_values(X_transformed)
    
    # Mean Absolute SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_imp_df = pd.DataFrame({"Feature": feature_names, "Mean_Abs_SHAP": mean_shap})
    shap_imp_df = shap_imp_df.sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    
    csv_path = os.path.join(OUTPUT_DIR, "02_shap_feature_importance.csv")
    shap_imp_df.to_csv(csv_path, index=False)
    log.info("Saved SHAP global importance: %s", csv_path)
    
    # ---------------------------------------------------------
    # 3. SHAP Visualizations
    # ---------------------------------------------------------
    log.info("Generating SHAP visualizations...")
    
    # a) Summary Plot (Bar)
    plt.figure()
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Global Feature Importance (Top 20)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03a_shap_summary_bar.png"), dpi=200)
    plt.close()
    
    # b) Summary Plot (Beeswarm)
    plt.figure()
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False, max_display=20)
    plt.title("SHAP Impact on Model Output (Top 20 Features)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03b_shap_beeswarm.png"), dpi=200)
    plt.close()
    
    # c) Top Feature Dependence Plot (e.g. HeavyAtoms or MW if they are top)
    top_feature = shap_imp_df.iloc[0]["Feature"]
    top_idx = feature_names.index(top_feature)
    
    plt.figure()
    shap.dependence_plot(top_idx, shap_values, X_transformed, feature_names=feature_names, show=False)
    plt.title(f"SHAP Dependence Plot: {top_feature}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03c_shap_dependence_top1.png"), dpi=200)
    plt.close()

    log.info("=" * 75)
    log.info("  SHAP ANALYSIS COMPLETED SUCCESSFULLY!")
    log.info("=" * 75)

if __name__ == "__main__":
    run_shap_analysis()