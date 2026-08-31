"""
cross_validation_and_residuals.py
=================================
Cross-Validation of Predictive Estimates and Residual Analysis for ChemPyQSPR.

Performs rigorous Out-of-Fold (OOF) cross-validation on the complete featurized
dataset, computes comprehensive residual diagnostics (normality tests,
heteroscedasticity, skewness, kurtosis, extreme outliers), and evaluates
model performance across distinct chemical functional groups.

Outputs:
    - Diagnostic figures saved to: data/processed/residual_analysis/
        * actual_vs_predicted_cv.png
        * residuals_vs_fitted.png
        * residual_distribution.png
        * residual_qq_plot.png
        * residuals_by_functional_group.png
        * residual_scale_location.png
    - Structured CSV data:
        * cv_predictions_and_residuals.csv
        * top_residual_outliers.csv
        * functional_group_residual_metrics.csv
    - Summary metrics JSON:
        * cv_residual_summary.json

Usage:
    python src/cross_validation_and_residuals.py
    python src/cross_validation_and_residuals.py --model-type svr --n-splits 5
    python src/cross_validation_and_residuals.py --model-path models/best_model.pkl
"""

# ----------------------------------------------------------------------
# Imports & Configuration
# ----------------------------------------------------------------------
import os
import sys
import json
import pickle
import logging
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, median_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# Configure Terminal Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Paths & Default Parameters
# ----------------------------------------------------------------------
INPUT_CSV   = "data/processed/featurized_dataset.csv"
OUTPUT_DIR  = "data/processed/residual_analysis"
RANDOM_STATE = 42
DEFAULT_N_SPLITS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Seaborn and Matplotlib Aesthetics
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "sans-serif"
})


# ======================================================================
# Section 1 -- Data Loading & Feature Preparation
# ======================================================================

def load_and_prepare_data(input_csv: str = INPUT_CSV) -> tuple:
    """
    Loads featurized dataset and extracts feature matrix X, target y,
    continuous/fingerprint column partitions, and compound metadata.

    Returns
    -------
    X               : pd.DataFrame -- Feature matrix
    y               : np.ndarray   -- Flash point target values (°C)
    df_meta         : pd.DataFrame -- Metadata columns (name, smiles, functional groups)
    continuous_cols : list[str]    -- Continuous descriptor names
    fingerprint_cols: list[str]    -- Binary fingerprint column names
    """
    if not os.path.exists(input_csv):
        log.error("Input file '%s' not found.", input_csv)
        sys.exit(1)

    log.info("Loading featurized dataset from: %s", input_csv)
    df = pd.read_csv(input_csv)

    target_col = "flash_point"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {input_csv}")

    meta_cols = ["name", "smiles", "primary_functional_group", "all_functional_groups", "cid", "cas"]
    existing_meta_cols = [c for c in meta_cols if c in df.columns]

    feature_cols = [c for c in df.columns if c not in meta_cols and c != target_col]

    # Clean numerical rows
    valid_mask = pd.to_numeric(df[target_col], errors="coerce").notna()
    df_clean = df[valid_mask].copy()
    df_clean[target_col] = pd.to_numeric(df_clean[target_col])

    X = df_clean[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df_clean[target_col].values
    df_meta = df_clean[existing_meta_cols].copy()

    continuous_cols  = [c for c in feature_cols if not c.startswith("fp_")]
    fingerprint_cols = [c for c in feature_cols if c.startswith("fp_")]

    log.info("Dataset successfully loaded: N = %d compounds.", len(df_clean))
    log.info("Features: %d continuous descriptors | %d Morgan fingerprint bits.",
             len(continuous_cols), len(fingerprint_cols))
    log.info("Target Flash Point: Min=%.2f deg C | Mean=%.2f deg C | Max=%.2f deg C | Std=%.2f deg C",
             y.min(), y.mean(), y.max(), y.std())

    return X, y, df_meta, continuous_cols, fingerprint_cols


# ======================================================================
# Section 2 -- Pipeline & Model Builder
# ======================================================================

def build_model_pipeline(
    model_type: str,
    continuous_cols: list,
    fingerprint_cols: list,
    use_variance_filter: bool = True
) -> Pipeline:
    """
    Builds a robust scikit-learn Pipeline with appropriate selective scaling
    and optional fingerprint dimensionality reduction (VarianceThreshold).
    """
    transformers = [
        ("num_scaler", StandardScaler(), continuous_cols)
    ]
    if use_variance_filter and len(fingerprint_cols) > 0:
        transformers.append(
            ("fp_selector", VarianceThreshold(threshold=0.01), fingerprint_cols)
        )
    elif len(fingerprint_cols) > 0:
        transformers.append(
            ("fp_passthrough", "passthrough", fingerprint_cols)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    model_type_norm = model_type.lower().strip()

    if model_type_norm in ["svr", "support_vector_regressor"]:
        estimator = SVR(kernel="rbf", C=100.0, epsilon=0.1, gamma="scale")
        model_name = "Support Vector Regressor (SVR - C=100, eps=0.1)"
    elif model_type_norm in ["xgb", "xgboost"]:
        estimator = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            reg_lambda=5.0,
            subsample=0.7,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        )
        model_name = "XGBoost Regressor (Opt params)"
    elif model_type_norm in ["rf", "random_forest"]:
        estimator = RandomForestRegressor(
            n_estimators=150,
            max_depth=6,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features=0.6,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model_name = "Random Forest Regressor (Opt params)"
    elif model_type_norm in ["ridge", "ridge_regression"]:
        estimator = Ridge(alpha=50.0, random_state=RANDOM_STATE)
        model_name = "Ridge Regression (alpha=50.0)"
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: 'svr', 'xgb', 'rf', 'ridge'")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])

    return pipeline, model_name


# ======================================================================
# Section 3 -- Out-of-Fold Cross-Validation Execution
# ======================================================================

def run_oof_cross_validation(
    pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = RANDOM_STATE
) -> tuple:
    """
    Executes K-Fold cross-validation generating out-of-fold predictions for every
    sample in the dataset, ensuring zero data leakage.

    Returns
    -------
    y_oof_pred    : np.ndarray -- Out-of-fold predicted flash points (°C)
    fold_indices  : np.ndarray -- Fold ID assigned to each sample (0 to K-1)
    fold_metrics  : list[dict] -- Metrics per validation fold
    """
    log.info("-" * 75)
    log.info("Starting %d-Fold Cross-Validation (Out-of-Fold Prediction Generation)...", n_splits)
    log.info("-" * 75)

    # Stratify fold generation by target quartiles for homogeneous distribution
    bins = pd.qcut(y, q=4, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y_oof_pred = np.zeros_like(y, dtype=float)
    fold_indices = np.zeros_like(y, dtype=int)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, bins)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        # Fit on training split
        pipeline.fit(X_tr, y_tr)

        # Predict on holdout validation split
        val_pred = pipeline.predict(X_val)
        y_oof_pred[val_idx] = val_pred
        fold_indices[val_idx] = fold

        r2_f = r2_score(y_val, val_pred)
        mae_f = mean_absolute_error(y_val, val_pred)
        rmse_f = root_mean_squared_error(y_val, val_pred)

        fold_metrics.append({
            "Fold": fold + 1,
            "N_Val": len(val_idx),
            "R2": round(r2_f, 4),
            "MAE_degC": round(mae_f, 2),
            "RMSE_degC": round(rmse_f, 2)
        })

        log.info("   Fold %d/%d - Val R2: %.4f | MAE: %.2f deg C | RMSE: %.2f deg C (n=%d)",
                 fold + 1, n_splits, r2_f, mae_f, rmse_f, len(val_idx))

    return y_oof_pred, fold_indices, fold_metrics


# ======================================================================
# Section 4 -- Comprehensive Residual Analysis & Diagnostics
# ======================================================================

def compute_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df_meta: pd.DataFrame
) -> tuple:
    """
    Computes residuals, absolute errors, percentage errors, normality tests,
    heteroscedasticity correlation, and subgroup performance.
    """
    residuals = y_true - y_pred                    # e_i = y_i - \hat{y}_i
    abs_errors = np.abs(residuals)                 # |e_i|
    std_residuals = residuals / np.std(residuals)  # Standardized residuals

    # 1. Overall Global Metrics
    r2_overall   = r2_score(y_true, y_pred)
    mae_overall  = mean_absolute_error(y_true, y_pred)
    rmse_overall = root_mean_squared_error(y_true, y_pred)
    medae_overall = median_absolute_error(y_true, y_pred)
    max_error    = np.max(abs_errors)

    # 2. Residual Distribution Statistics
    res_mean = np.mean(residuals)
    res_std  = np.std(residuals)
    res_skew = stats.skew(residuals)
    res_kurt = stats.kurtosis(residuals)

    # 3. Normality Tests on Residuals
    # Jarque-Bera (tests skewness and kurtosis matching normal distribution)
    jb_stat, jb_pvalue = stats.jarque_bera(residuals)

    # Shapiro-Wilk test on representative random subsample (n=5000 max for scipy)
    sample_size = min(len(residuals), 5000)
    np.random.seed(RANDOM_STATE)
    res_subsample = np.random.choice(residuals, size=sample_size, replace=False)
    sw_stat, sw_pvalue = stats.shapiro(res_subsample)

    # 4. Heteroscedasticity Check: Spearman rank correlation between |residuals| and fitted values
    spearman_corr, spearman_pvalue = stats.spearmanr(y_pred, abs_errors)

    # Compile Summary Dictionary
    summary_metrics = {
        "N_Samples": int(len(y_true)),
        "CV_R2": round(float(r2_overall), 4),
        "CV_MAE_degC": round(float(mae_overall), 2),
        "CV_RMSE_degC": round(float(rmse_overall), 2),
        "CV_Median_AE_degC": round(float(medae_overall), 2),
        "Max_Absolute_Error_degC": round(float(max_error), 2),
        "Residual_Mean_Bias_degC": round(float(res_mean), 3),
        "Residual_StdDev_degC": round(float(res_std), 2),
        "Residual_Skewness": round(float(res_skew), 3),
        "Residual_Excess_Kurtosis": round(float(res_kurt), 3),
        "Jarque_Bera_Statistic": round(float(jb_stat), 2),
        "Jarque_Bera_pvalue": float(jb_pvalue),
        "Shapiro_Wilk_Statistic": round(float(sw_stat), 4),
        "Shapiro_Wilk_pvalue": float(sw_pvalue),
        "Spearman_Heteroscedasticity_Corr": round(float(spearman_corr), 4),
        "Spearman_Heteroscedasticity_pvalue": float(spearman_pvalue)
    }

    # 5. Build Enriched Results DataFrame
    results_df = df_meta.copy()
    results_df["actual_flash_point"] = y_true
    results_df["cv_predicted_flash_point"] = np.round(y_pred, 2)
    results_df["residual"] = np.round(residuals, 2)
    results_df["abs_error"] = np.round(abs_errors, 2)
    results_df["std_residual"] = np.round(std_residuals, 2)

    # Outlier Flag: |std_residual| > 3.0
    results_df["is_extreme_outlier"] = (np.abs(std_residuals) > 3.0)

    # 6. Functional Group Subgroup Performance
    fg_col = "primary_functional_group"
    if fg_col in results_df.columns:
        fg_metrics = []
        for fg, group_df in results_df.groupby(fg_col):
            if len(group_df) < 5:
                continue
            fg_y_true = group_df["actual_flash_point"].values
            fg_y_pred = group_df["cv_predicted_flash_point"].values
            fg_res    = group_df["residual"].values

            fg_metrics.append({
                "Functional_Group": fg,
                "Count": len(group_df),
                "R2": round(r2_score(fg_y_true, fg_y_pred), 4),
                "MAE (°C)": round(mean_absolute_error(fg_y_true, fg_y_pred), 2),
                "RMSE (°C)": round(root_mean_squared_error(fg_y_true, fg_y_pred), 2),
                "Mean_Bias (°C)": round(float(np.mean(fg_res)), 2),
                "Outliers (>3σ)": int(group_df["is_extreme_outlier"].sum())
            })
        fg_df = pd.DataFrame(fg_metrics).sort_values(by="Count", ascending=False).reset_index(drop=True)
    else:
        fg_df = pd.DataFrame()

    return summary_metrics, results_df, fg_df


# ======================================================================
# Section 5 -- Diagnostic Visualizations
# ======================================================================

def generate_diagnostic_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    results_df: pd.DataFrame,
    model_name: str,
    output_dir: str = OUTPUT_DIR
) -> None:
    """
    Generates 6 publication-quality statistical diagnostic charts.
    """
    log.info("-" * 75)
    log.info("Generating Publication-Quality Diagnostic Figures in: %s", output_dir)
    log.info("-" * 75)

    r2_val = r2_score(y_true, y_pred)
    rmse_val = root_mean_squared_error(y_true, y_pred)
    mae_val = mean_absolute_error(y_true, y_pred)

    # ------------------------------------------------------------------
    # Plot 1: Actual vs. Predicted (Cross-Validated)
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 7))
    plt.scatter(y_true, y_pred, alpha=0.35, edgecolors="none", s=28, color="#1f77b4", label="Compounds (OOF CV)")
    
    # 1:1 Reference Line
    min_val = min(y_true.min(), y_pred.min()) - 10
    max_val = max(y_true.max(), y_pred.max()) + 10
    plt.plot([min_val, max_val], [min_val, max_val], color="#d62728", linestyle="--", linewidth=2, label="Perfect Agreement (1:1)")

    # Linear Regression Fit
    m, b = np.polyfit(y_true, y_pred, 1)
    plt.plot(np.array([min_val, max_val]), m * np.array([min_val, max_val]) + b, color="#2ca02c", linestyle="-", linewidth=1.8, label=f"Fit (y = {m:.2f}x + {b:.1f})")

    plt.title(f"Cross-Validated Actual vs. Predicted Flash Point\nModel: {model_name}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Actual Experimental Flash Point (°C)", fontsize=11)
    plt.ylabel("Cross-Validated Predicted Flash Point (°C)", fontsize=11)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    # Metric text box
    stats_text = f"N = {len(y_true):,}\n$R^2$ = {r2_val:.4f}\nMAE = {mae_val:.2f} °C\nRMSE = {rmse_val:.2f} °C"
    plt.gca().text(0.05, 0.92, stats_text, transform=plt.gca().transAxes, fontsize=11,
                   verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9))

    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    p1_path = os.path.join(output_dir, "actual_vs_predicted_cv.png")
    plt.savefig(p1_path)
    plt.close()
    log.info("   Saved: %s", p1_path)

    # ------------------------------------------------------------------
    # Plot 2: Residuals vs. Fitted Values (Homoscedasticity & Linearity)
    # ------------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    plt.scatter(y_pred, residuals, alpha=0.35, color="#34495e", s=25, label=r"Residuals ($y - \hat{y}$)")
    plt.axhline(0, color="#e74c3c", linestyle="--", linewidth=1.8, label="Zero Bias Baseline ($e=0$)")

    # 3-Sigma Bounds
    std_res = np.std(residuals)
    plt.axhline(3 * std_res, color="#e67e22", linestyle=":", linewidth=1.5, label="+3σ Limit")
    plt.axhline(-3 * std_res, color="#e67e22", linestyle=":", linewidth=1.5, label="-3σ Limit")

    # Trend moving average
    sorted_idx = np.argsort(y_pred)
    sorted_pred = y_pred[sorted_idx]
    sorted_res = residuals[sorted_idx]
    rolling_window = max(20, len(y_pred) // 40)
    rolling_mean = pd.Series(sorted_res).rolling(window=rolling_window, min_periods=10, center=True).mean()
    plt.plot(sorted_pred, rolling_mean, color="#2980b9", linewidth=2.5, label=f"Trend Moving Avg (w={rolling_window})")

    plt.title("Residuals vs. Fitted Values (Homoscedasticity Diagnostic)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Cross-Validated Fitted Flash Point (°C)", fontsize=11)
    plt.ylabel("Residual: Experimental - Predicted (°C)", fontsize=11)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    p2_path = os.path.join(output_dir, "residuals_vs_fitted.png")
    plt.savefig(p2_path)
    plt.close()
    log.info("   Saved: %s", p2_path)

    # ------------------------------------------------------------------
    # Plot 3: Residual Distribution (Histogram + KDE + Gaussian Fit)
    # ------------------------------------------------------------------
    fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [0.2, 0.8]})

    # Boxplot on top
    sns.boxplot(x=residuals, ax=ax_box, color="#5dade2", fliersize=3)
    ax_box.set(xlabel="")
    ax_box.set_title("Distribution of Cross-Validation Residuals", fontsize=13, fontweight="bold", pad=12)

    # Histogram & KDE
    sns.histplot(residuals, kde=True, ax=ax_hist, color="#2e86c1", stat="density", bins=60, alpha=0.5, label="Empirical KDE")

    # Theoretical Normal Curve
    x_axis = np.linspace(residuals.min(), residuals.max(), 300)
    pdf_fitted = stats.norm.pdf(x_axis, loc=np.mean(residuals), scale=np.std(residuals))
    ax_hist.plot(x_axis, pdf_fitted, "r--", linewidth=2, label=rf"Normal Fit ($\mu$={np.mean(residuals):.1f}, $\sigma$={np.std(residuals):.1f})")

    ax_hist.axvline(0, color="black", linestyle=":", linewidth=1.2)
    ax_hist.set_xlabel("Residual: Experimental - Predicted (°C)", fontsize=11)
    ax_hist.set_ylabel("Probability Density", fontsize=11)
    ax_hist.legend(loc="upper right")

    plt.tight_layout()
    p3_path = os.path.join(output_dir, "residual_distribution.png")
    plt.savefig(p3_path)
    plt.close()
    log.info("   Saved: %s", p3_path)

    # ------------------------------------------------------------------
    # Plot 4: Normal Q-Q Plot
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 7))
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm", fit=True)
    plt.scatter(osm, osr, alpha=0.45, color="#16a085", s=25, label="Empirical Residuals")
    plt.plot(osm, slope * np.array(osm) + intercept, color="#c0392b", linestyle="--", linewidth=2, label=f"Theoretical Line ($R^2$={r**2:.4f})")

    plt.title("Normal Q-Q Plot of Model Residuals", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Theoretical Standard Normal Quantiles", fontsize=11)
    plt.ylabel("Ordered Sample Residuals (°C)", fontsize=11)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    p4_path = os.path.join(output_dir, "residual_qq_plot.png")
    plt.savefig(p4_path)
    plt.close()
    log.info("   Saved: %s", p4_path)

    # ------------------------------------------------------------------
    # Plot 5: Residuals by Functional Group
    # ------------------------------------------------------------------
    if "primary_functional_group" in results_df.columns:
        # Keep top 12 most abundant groups for clarity
        top_groups = results_df["primary_functional_group"].value_counts().head(12).index.tolist()
        df_top_fg = results_df[results_df["primary_functional_group"].isin(top_groups)].copy()

        plt.figure(figsize=(12, 6))
        group_order = df_top_fg.groupby("primary_functional_group")["residual"].median().sort_values().index

        sns.boxplot(
            data=df_top_fg,
            x="primary_functional_group",
            y="residual",
            order=group_order,
            palette="Spectral",
            fliersize=2.5
        )
        plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
        plt.title("Residual Distribution Across Top Chemical Functional Groups", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Primary Functional Group", fontsize=11)
        plt.ylabel("Residual (°C)", fontsize=11)
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        p5_path = os.path.join(output_dir, "residuals_by_functional_group.png")
        plt.savefig(p5_path)
        plt.close()
        log.info("   Saved: %s", p5_path)

    # ------------------------------------------------------------------
    # Plot 6: Scale-Location Plot (Spread-Location)
    # ------------------------------------------------------------------
    std_residuals = residuals / np.std(residuals)
    sqrt_abs_std_res = np.sqrt(np.abs(std_residuals))

    plt.figure(figsize=(9, 6))
    plt.scatter(y_pred, sqrt_abs_std_res, alpha=0.35, color="#8e44ad", s=25)

    # Smooth trendline
    sorted_idx = np.argsort(y_pred)
    sorted_pred = y_pred[sorted_idx]
    sorted_sqrt = sqrt_abs_std_res[sorted_idx]
    rolling_sqrt = pd.Series(sorted_sqrt).rolling(window=rolling_window, min_periods=10, center=True).mean()
    plt.plot(sorted_pred, rolling_sqrt, color="#e74c3c", linewidth=2.5, label="Spread Trendline")

    plt.title("Scale-Location Plot (Homoscedasticity Assessment)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Cross-Validated Fitted Flash Point (°C)", fontsize=11)
    plt.ylabel(r"$\sqrt{|\mathrm{Standardized\ Residuals}|}$", fontsize=11)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    p6_path = os.path.join(output_dir, "residual_scale_location.png")
    plt.savefig(p6_path)
    plt.close()
    log.info("   Saved: %s", p6_path)


# ======================================================================
# Section 6 -- CLI & Orchestrator
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="ChemPyQSPR -- Cross-Validation & Residual Analysis Pipeline"
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=INPUT_CSV,
        help=f"Path to featurized dataset. Default: {INPUT_CSV}"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="svr",
        choices=["svr", "xgb", "rf", "ridge"],
        help="Algorithm to cross-validate. Default: 'svr' (champion model from grid search)"
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help=f"Number of K-Fold CV splits. Default: {DEFAULT_N_SPLITS}"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output folder for figures and CSVs. Default: {OUTPUT_DIR}"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("=" * 75)
    log.info(" ChemPyQSPR -- Out-of-Fold Cross-Validation & Residual Analysis")
    log.info(" Timestamp : %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info(" Selected Model Type : %s", args.model_type.upper())
    log.info("=" * 75)

    # 1. Load Data
    X, y, df_meta, continuous_cols, fingerprint_cols = load_and_prepare_data(args.input_csv)

    # 2. Build Model Pipeline
    pipeline, model_name = build_model_pipeline(
        model_type=args.model_type,
        continuous_cols=continuous_cols,
        fingerprint_cols=fingerprint_cols,
        use_variance_filter=True
    )
    log.info("Constructed Predictive Pipeline: %s", model_name)

    # 3. Execute Out-of-Fold Cross-Validation
    y_oof_pred, fold_indices, fold_metrics = run_oof_cross_validation(
        pipeline=pipeline,
        X=X,
        y=y,
        n_splits=args.n_splits,
        random_state=RANDOM_STATE
    )

    # 4. Compute Residual Diagnostics
    summary_metrics, results_df, fg_df = compute_residual_diagnostics(
        y_true=y,
        y_pred=y_oof_pred,
        df_meta=df_meta
    )
    summary_metrics["model_name"] = model_name
    summary_metrics["timestamp"] = datetime.now().isoformat()
    summary_metrics["fold_metrics"] = fold_metrics

    # 5. Export Datasets & Summaries
    pred_csv_path = os.path.join(args.output_dir, "cv_predictions_and_residuals.csv")
    results_df.to_csv(pred_csv_path, index=False)
    log.info("Exported full OOF predictions & residuals to: %s", pred_csv_path)

    # Extreme Outliers CSV (top 30 largest absolute errors)
    outliers_df = results_df[results_df["is_extreme_outlier"]].sort_values(by="abs_error", ascending=False)
    outliers_csv_path = os.path.join(args.output_dir, "top_residual_outliers.csv")
    outliers_df.to_csv(outliers_csv_path, index=False)
    log.info("Exported %d extreme outliers (|z| > 3 sigma) to: %s", len(outliers_df), outliers_csv_path)

    # Functional Group Metrics CSV
    if not fg_df.empty:
        fg_csv_path = os.path.join(args.output_dir, "functional_group_residual_metrics.csv")
        fg_df.to_csv(fg_csv_path, index=False)
        log.info("Exported functional group residual breakdown to: %s", fg_csv_path)

    # JSON Summary
    summary_json_path = os.path.join(args.output_dir, "cv_residual_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2, default=str)
    log.info("Exported summary metrics JSON to: %s", summary_json_path)

    # 6. Generate Publication Figures
    generate_diagnostic_plots(
        y_true=y,
        y_pred=y_oof_pred,
        residuals=results_df["residual"].values,
        results_df=results_df,
        model_name=model_name,
        output_dir=args.output_dir
    )

    # 7. Print Comprehensive Summary
    log.info("=" * 75)
    log.info("CROSS-VALIDATION & RESIDUAL ANALYSIS COMPLETED SUCCESSFULLY!")
    log.info("=" * 75)
    log.info("OVERALL CV PERFORMANCE:")
    log.info("  * Out-of-Fold R2       : %.4f", summary_metrics["CV_R2"])
    log.info("  * Out-of-Fold MAE      : %.2f deg C", summary_metrics["CV_MAE_degC"])
    log.info("  * Out-of-Fold RMSE     : %.2f deg C", summary_metrics["CV_RMSE_degC"])
    log.info("  * Out-of-Fold Median AE: %.2f deg C", summary_metrics["CV_Median_AE_degC"])
    log.info("  * Max Absolute Error   : %.2f deg C", summary_metrics["Max_Absolute_Error_degC"])
    log.info("-" * 75)
    log.info("RESIDUAL DISTRIBUTION PROPERTIES:")
    log.info("  * Mean Residual (Bias) : %.3f deg C (Ideal: 0.0)", summary_metrics["Residual_Mean_Bias_degC"])
    log.info("  * Residual Std Dev     : %.2f deg C", summary_metrics["Residual_StdDev_degC"])
    log.info("  * Residual Skewness    : %.3f", summary_metrics["Residual_Skewness"])
    log.info("  * Residual Kurtosis    : %.3f", summary_metrics["Residual_Excess_Kurtosis"])
    log.info("  * Extreme Outliers     : %d / %d (%.2f%%)",
             len(outliers_df), len(y), (len(outliers_df) / len(y)) * 100)
    log.info("-" * 75)
    if not fg_df.empty:
        log.info("TOP FUNCTIONAL GROUPS BY MAE:\n\n%s\n",
                 fg_df[["Functional_Group", "Count", "MAE (°C)", "RMSE (°C)", "Mean_Bias (°C)", "R2"]].head(8).to_string(index=False))
    log.info("All deliverables saved in: %s", args.output_dir)
    log.info("=" * 75)


if __name__ == "__main__":
    main()
