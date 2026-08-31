"""
fingerprint_analysis.py
=======================
Morgan Fingerprint Analysis and Dimensionality Reduction for ChemPyQSPR.

This script analyzes the 1024-bit Morgan Fingerprints generated during the
featurization step. It performs the following analyses:
    1. Bit Frequency & Low-Variance Analysis -- Identifies near-constant bits
    2. Principal Component Analysis (PCA)    -- Linear dimensionality reduction
    3. t-Distributed Stochastic Neighbor Embedding (t-SNE) -- Non-linear clustering
    4. Point-Biserial Correlation            -- Identifies bits most correlated with flash point

Outputs:
    All figures are saved to: data/processed/fingerprint_results/
    All summary tables are exported to CSV in the same folder.

Usage:
    Run from the project root:
        python src/fingerprint_analysis.py
"""

# ----------------------------------------------------------------------
# Imports & Configuration
# ----------------------------------------------------------------------
import os
import sys
import warnings
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

# Configure terminal logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Paths & Plot Style
# ----------------------------------------------------------------------
INPUT_CSV  = "data/processed/featurized_dataset.csv"
OUTPUT_DIR = "data/processed/fingerprint_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL = "flash_point"
GROUP_COL  = "primary_functional_group"

# Consistent color palette (one color per functional group)
FG_PALETTE = {
    "hydrocarbon":    "#4C8EDA",
    "amine":          "#E07B39",
    "alcohol":        "#3BAD7A",
    "ester":          "#D94F5C",
    "organohalide":   "#9B59B6",
    "aromatic":       "#F0C419",
    "ketone":         "#1ABC9C",
    "ether":          "#E74C3C",
    "carboxylic_acid":"#2980B9",
    "aldehyde":       "#8E44AD",
    "nitrile":        "#27AE60",
    "nitro":          "#E67E22",
}

# Global Seaborn / Matplotlib defaults
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi":   150,
    "savefig.dpi":  200,
    "savefig.bbox": "tight",
    "font.family":  "DejaVu Sans",
})

# ======================================================================
# HELPER UTILITIES
# ======================================================================

def save_fig(filename: str):
    """Saves the current Matplotlib figure to the output directory."""
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path)
    plt.close()
    log.info("   Figure saved: %s", path)

def section_header(title: str):
    """Prints a formatted section header to the terminal."""
    log.info("")
    log.info("=" * 75)
    log.info("  %s", title)
    log.info("=" * 75)

# ======================================================================
# 1. LOAD DATASET
# ======================================================================

def load_dataset(path: str) -> pd.DataFrame:
    """Loads and performs a basic integrity check on the input CSV."""
    if not os.path.exists(path):
        log.error("Input file not found: %s", path)
        sys.exit(1)

    df = pd.read_csv(path)
    log.info("Dataset loaded: %d compounds | %d columns.", len(df), len(df.columns))
    
    # Coerce flash point to numeric
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL]).copy()
    return df

# ======================================================================
# 2. BIT FREQUENCY ANALYSIS
# ======================================================================

def bit_frequency_analysis(df: pd.DataFrame) -> list:
    """
    Analyzes the frequency of 1s in each bit.
    Identifies bits with near-zero variance (present in <1% or >99% of compounds).
    """
    section_header("SECTION 1 -- BIT FREQUENCY & VARIANCE ANALYSIS")

    fp_cols = [c for c in df.columns if c.startswith("fp_")]
    if not fp_cols:
        log.error("No fingerprint columns (starting with 'fp_') found.")
        sys.exit(1)

    log.info("Analyzing %d fingerprint bits...", len(fp_cols))

    fp_data = df[fp_cols]
    
    # Calculate frequency (mean of binary column)
    frequencies = fp_data.mean()
    
    # Identify near-zero variance bits
    threshold = 0.01 # 1%
    low_var_mask = (frequencies < threshold) | (frequencies > (1 - threshold))
    low_var_cols = frequencies[low_var_mask].index.tolist()
    
    log.info("Total bits: %d", len(fp_cols))
    log.info("Bits with near-zero variance (<1%% or >99%%): %d", len(low_var_cols))
    log.info("Informative bits remaining: %d", len(fp_cols) - len(low_var_cols))

    # --- Plot distribution of bit frequencies ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(frequencies * 100, bins=50, color="#4C8EDA", edgecolor="white")
    ax.axvline(threshold * 100, color="#D94F5C", linestyle="--", label=f"{threshold*100}% Threshold")
    ax.axvline((1-threshold) * 100, color="#D94F5C", linestyle="--")
    ax.set_xlabel("Percentage of Compounds with Bit = 1")
    ax.set_ylabel("Number of Bits")
    ax.set_title("Distribution of Fingerprint Bit Frequencies", fontweight="bold")
    ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    save_fig("01_bit_frequency_distribution.png")

    informative_cols = [c for c in fp_cols if c not in low_var_cols]
    
    # Export bit frequencies
    freq_df = pd.DataFrame({"Bit": frequencies.index, "Frequency": frequencies.values, "Is_Low_Variance": low_var_mask.values})
    csv_path = os.path.join(OUTPUT_DIR, "01_bit_frequencies.csv")
    freq_df.to_csv(csv_path, index=False)
    log.info("   Bit frequencies saved: %s", csv_path)

    return informative_cols

# ======================================================================
# 3. PCA DIMENSIONALITY REDUCTION
# ======================================================================

def perform_pca(df: pd.DataFrame, fp_cols: list):
    """
    Performs Principal Component Analysis on the informative fingerprint bits.
    """
    section_header("SECTION 2 -- PRINCIPAL COMPONENT ANALYSIS (PCA)")

    fp_data = df[fp_cols]
    
    pca = PCA(n_components=10)
    pca_result = pca.fit_transform(fp_data)
    
    explained_variance = pca.explained_variance_ratio_ * 100
    log.info("Variance explained by PC1: %.2f%%", explained_variance[0])
    log.info("Variance explained by PC2: %.2f%%", explained_variance[1])
    log.info("Cumulative variance explained by top 10 PCs: %.2f%%", sum(explained_variance))

    df_pca = pd.DataFrame({"PC1": pca_result[:, 0], "PC2": pca_result[:, 1]})
    df_pca[TARGET_COL] = df[TARGET_COL].values
    
    has_group = GROUP_COL in df.columns
    if has_group:
        df_pca[GROUP_COL] = df[GROUP_COL].values

    # --- Plot PCA Colored by Flash Point ---
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(df_pca["PC1"], df_pca["PC2"], c=df_pca[TARGET_COL], cmap="coolwarm", s=10, alpha=0.6)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Flash Point (deg C)")
    ax.set_xlabel(f"Principal Component 1 ({explained_variance[0]:.1f}%)")
    ax.set_ylabel(f"Principal Component 2 ({explained_variance[1]:.1f}%)")
    ax.set_title("PCA of Morgan Fingerprints (Colored by Flash Point)", fontweight="bold")
    plt.tight_layout()
    save_fig("02a_pca_flash_point.png")

    # --- Plot PCA Colored by Functional Group ---
    if has_group:
        fig, ax = plt.subplots(figsize=(10, 7))
        for group, gdf in df_pca.groupby(GROUP_COL):
            color = FG_PALETTE.get(group, "#888888")
            ax.scatter(gdf["PC1"], gdf["PC2"], s=10, alpha=0.6, color=color, label=group)
        ax.set_xlabel(f"Principal Component 1 ({explained_variance[0]:.1f}%)")
        ax.set_ylabel(f"Principal Component 2 ({explained_variance[1]:.1f}%)")
        ax.set_title("PCA of Morgan Fingerprints (Colored by Functional Group)", fontweight="bold")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=2)
        plt.tight_layout()
        save_fig("02b_pca_functional_groups.png")
        
    return pca, df_pca

# ======================================================================
# 4. t-SNE CLUSTERING
# ======================================================================

def perform_tsne(df: pd.DataFrame, fp_cols: list):
    """
    Performs t-SNE non-linear dimensionality reduction on the fingerprint bits.
    """
    section_header("SECTION 3 -- t-SNE CLUSTERING")

    fp_data = df[fp_cols]
    
    # Use top 50 PCs to speed up t-SNE
    pca = PCA(n_components=min(50, len(fp_cols)))
    fp_pca = pca.fit_transform(fp_data)
    
    log.info("Running t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    tsne_result = tsne.fit_transform(fp_pca)
    
    df_tsne = pd.DataFrame({"tSNE1": tsne_result[:, 0], "tSNE2": tsne_result[:, 1]})
    df_tsne[TARGET_COL] = df[TARGET_COL].values
    
    has_group = GROUP_COL in df.columns
    if has_group:
        df_tsne[GROUP_COL] = df[GROUP_COL].values

    # --- Plot t-SNE Colored by Flash Point ---
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(df_tsne["tSNE1"], df_tsne["tSNE2"], c=df_tsne[TARGET_COL], cmap="coolwarm", s=8, alpha=0.6)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Flash Point (deg C)")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title("t-SNE of Morgan Fingerprints (Colored by Flash Point)", fontweight="bold")
    plt.tight_layout()
    save_fig("03a_tsne_flash_point.png")

    # --- Plot t-SNE Colored by Functional Group ---
    if has_group:
        fig, ax = plt.subplots(figsize=(10, 7))
        for group, gdf in df_tsne.groupby(GROUP_COL):
            color = FG_PALETTE.get(group, "#888888")
            ax.scatter(gdf["tSNE1"], gdf["tSNE2"], s=8, alpha=0.6, color=color, label=group)
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_title("t-SNE of Morgan Fingerprints (Colored by Functional Group)", fontweight="bold")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=2)
        plt.tight_layout()
        save_fig("03b_tsne_functional_groups.png")
        
    return df_tsne

# ======================================================================
# 5. POINT-BISERIAL CORRELATION
# ======================================================================

def point_biserial_correlation(df: pd.DataFrame, fp_cols: list):
    """
    Computes point-biserial correlation for each bit vs flash point.
    """
    section_header("SECTION 4 -- POINT-BISERIAL CORRELATION")

    y = df[TARGET_COL].values
    fp_data = df[fp_cols].values
    
    corrs = []
    p_values = []
    
    log.info("Calculating point-biserial correlations for %d bits...", len(fp_cols))
    for i in range(fp_data.shape[1]):
        r, p = pointbiserialr(fp_data[:, i], y)
        corrs.append(r)
        p_values.append(p)
        
    results_df = pd.DataFrame({
        "Bit": fp_cols,
        "Correlation": corrs,
        "P_Value": p_values
    })
    
    # Sort by absolute correlation
    results_df["Abs_Corr"] = results_df["Correlation"].abs()
    results_df = results_df.sort_values(by="Abs_Corr", ascending=False).reset_index(drop=True)
    
    top20 = results_df.head(20)
    log.info("\nTop 10 Bits most correlated with Flash Point:")
    log.info("\n%s", top20.head(10).to_string(index=False, columns=["Bit", "Correlation", "P_Value"]))
    
    # --- Plot Top 20 Correlated Bits ---
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4C8EDA" if val > 0 else "#D94F5C" for val in top20["Correlation"]]
    bars = ax.bar(top20["Bit"], top20["Correlation"], color=colors, edgecolor="white")
    ax.axhline(0, color="gray", lw=1)
    ax.set_xticklabels(top20["Bit"], rotation=45, ha="right")
    ax.set_xlabel("Morgan Fingerprint Bit")
    ax.set_ylabel("Point-Biserial Correlation (r)")
    ax.set_title("Top 20 Fingerprint Bits Correlated with Flash Point", fontweight="bold")
    plt.tight_layout()
    save_fig("04_top_correlated_bits.png")
    
    # Export results
    csv_path = os.path.join(OUTPUT_DIR, "04_bit_correlations.csv")
    results_df.to_csv(csv_path, index=False)
    log.info("   Correlation table saved: %s", csv_path)

# ======================================================================
# MAIN PIPELINE
# ======================================================================

def run_pipeline():
    log.info("=" * 75)
    log.info("  ChemPyQSPR -- Morgan Fingerprint Analysis")
    log.info("=" * 75)
    
    df = load_dataset(INPUT_CSV)
    
    # 1. Bit Frequency Analysis
    informative_cols = bit_frequency_analysis(df)
    
    # 2. PCA
    perform_pca(df, informative_cols)
    
    # 3. t-SNE
    perform_tsne(df, informative_cols)
    
    # 4. Point-Biserial Correlation
    point_biserial_correlation(df, informative_cols)
    
    log.info("=" * 75)
    log.info("  FINGERPRINT ANALYSIS COMPLETED SUCCESSFULLY!")
    log.info("=" * 75)

if __name__ == "__main__":
    run_pipeline()