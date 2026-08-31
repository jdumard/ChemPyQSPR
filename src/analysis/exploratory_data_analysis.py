"""
exploratory_data_analysis.py
============================
Exploratory Data Analysis (EDA) for the ChemPyQSPR Flash Point Prediction Project.

This script analyzes the enriched dataset (cleaned_flash_point_database_fg.csv) and
produces a comprehensive set of statistical and visual diagnostics, organized into
the following sections:

    1. Descriptive Statistics    -- Summary stats for flash point and RDKit descriptors
    2. Target Variable Analysis  -- Distribution, skewness, normality checks
    3. Functional Group Analysis -- Count, proportion, and flash point by group
    4. Outlier Assessment        -- IQR-based detection, z-score, and visual evaluation
    5. Correlation Analysis      -- Pearson correlation matrix for physicochemical features
    6. Scatter Plots             -- Descriptor vs. flash point (bivariate relationships)
    7. Descriptor Histograms     -- Per-descriptor distributions by functional group
    8. Suggested Next Analyses   -- Printed recommendations for further investigation

Note on Morgan Fingerprints:
    Morgan fingerprints (1024 binary bits) are intentionally excluded from the
    correlation and scatter-plot analyses.  Their high dimensionality and binary
    nature make standard correlation metrics uninformative.  Appropriate analyses
    for fingerprints (PCA, t-SNE, UMAP, point-biserial correlation) are listed
    in the "Suggested Next Analyses" section at the end of the script.

Outputs:
    All figures are saved to: data/processed/eda_results/
    All summary tables are exported to CSV in the same folder.

Usage:
    Run from the project root:
        python src/exploratory_data_analysis.py
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
from scipy import stats

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
INPUT_CSV  = "data/processed/cleaned_flash_point_database_fg.csv"
OUTPUT_DIR = "data/processed/eda_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# RDKit descriptor columns present in the featurized dataset
DESCRIPTOR_COLS = [
    "MW", "LogP", "TPSA", "HBD", "HBA",
    "RotatableBonds", "AromaticRings", "HeavyAtoms",
    "MolarRefractivity", "FractionCSP3"
]

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
        log.error("Run 'src/identify_funcional_groups.py' first to generate this file.")
        sys.exit(1)

    df = pd.read_csv(path)
    log.info("Dataset loaded: %d compounds | %d columns.", len(df), len(df.columns))
    log.info("Columns: %s", df.columns.tolist())

    # Coerce flash point to numeric (safety net)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[TARGET_COL]).copy()
    if len(df) < before:
        log.warning("Dropped %d rows with invalid flash point values.", before - len(df))

    return df


# ======================================================================
# 2. DESCRIPTIVE STATISTICS
# ======================================================================

def descriptive_statistics(df: pd.DataFrame):
    """
    Computes and exports descriptive statistics for the flash point
    target variable.  RDKit descriptor statistics are produced separately
    in Section 7 using the featurized dataset.
    """
    section_header("SECTION 1 -- DESCRIPTIVE STATISTICS")

    fp = df[TARGET_COL]

    stats_dict = {
        "Count":                    fp.count(),
        "Mean (deg C)":             round(fp.mean(), 3),
        "Median (deg C)":           round(fp.median(), 3),
        "Std Dev (deg C)":          round(fp.std(), 3),
        "Min (deg C)":              round(fp.min(), 3),
        "Max (deg C)":              round(fp.max(), 3),
        "Range (deg C)":            round(fp.max() - fp.min(), 3),
        "Q1 (25th pct.) (deg C)":   round(fp.quantile(0.25), 3),
        "Q3 (75th pct.) (deg C)":   round(fp.quantile(0.75), 3),
        "IQR (deg C)":              round(fp.quantile(0.75) - fp.quantile(0.25), 3),
        "Skewness":                 round(fp.skew(), 4),
        "Kurtosis (excess)":        round(fp.kurt(), 4),
        "CV (%)":                   round((fp.std() / fp.mean()) * 100, 3) if fp.mean() != 0 else float("nan"),
    }

    # Shapiro-Wilk on a random 5000-sample subset (computationally feasible)
    sample = fp.dropna().sample(min(5000, len(fp)), random_state=42)
    stat_sw, p_sw = stats.shapiro(sample)
    stats_dict["Shapiro-Wilk W (N<=5000)"] = round(float(stat_sw), 6)
    stats_dict["Shapiro-Wilk p-value"]      = round(float(p_sw), 6)

    stats_series = pd.Series(stats_dict, name="Flash Point")
    log.info("\n%s", stats_series.to_string())

    # Export
    csv_path = os.path.join(OUTPUT_DIR, "01_descriptive_statistics.csv")
    stats_series.to_frame().to_csv(csv_path)
    log.info("   Table saved: %s", csv_path)

    return stats_dict


# ======================================================================
# 3. TARGET VARIABLE DISTRIBUTION
# ======================================================================

def plot_target_distribution(df: pd.DataFrame):
    """
    Plots the flash point distribution using:
        - Histogram + KDE overlay
        - Q-Q plot (normal quantile-quantile)
        - Empirical Cumulative Distribution Function (ECDF)
    """
    section_header("SECTION 2 -- TARGET VARIABLE DISTRIBUTION")

    fp = df[TARGET_COL].dropna()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Flash Point Distribution Analysis", fontsize=15, fontweight="bold", y=1.02)

    # --- 2a. Histogram + KDE ---
    ax = axes[0]
    ax.hist(fp, bins=60, color="#4C8EDA", edgecolor="white", alpha=0.80,
            density=True, label="Histogram")
    kde_x = np.linspace(fp.min(), fp.max(), 400)
    kde = stats.gaussian_kde(fp)
    ax.plot(kde_x, kde(kde_x), color="#E07B39", lw=2.2, label="KDE")
    ax.axvline(fp.mean(),   color="#D94F5C", lw=1.8, linestyle="--",
               label=f"Mean = {fp.mean():.1f} C")
    ax.axvline(fp.median(), color="#3BAD7A", lw=1.8, linestyle=":",
               label=f"Median = {fp.median():.1f} C")
    ax.set_xlabel("Flash Point (deg C)")
    ax.set_ylabel("Density")
    ax.set_title("Histogram + KDE")
    ax.legend(fontsize=9)

    # --- 2b. Q-Q Plot ---
    ax = axes[1]
    (osm, osr), (slope, intercept, r) = stats.probplot(fp, dist="norm")
    ax.scatter(osm, osr, s=6, color="#4C8EDA", alpha=0.35, label="Observed quantiles")
    ax.plot(osm, slope * np.array(osm) + intercept, color="#E07B39", lw=2,
            label=f"Normal fit (R={r:.4f})")
    ax.set_xlabel("Theoretical Quantiles (Normal)")
    ax.set_ylabel("Sample Quantiles (deg C)")
    ax.set_title("Q-Q Plot vs. Normal Distribution")
    ax.legend(fontsize=9)

    # --- 2c. ECDF ---
    ax = axes[2]
    x_sorted = np.sort(fp)
    y_ecdf   = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    ax.step(x_sorted, y_ecdf, color="#4C8EDA", lw=1.8, label="ECDF")
    ax.axhline(0.25, color="#9B59B6", lw=1.2, linestyle="--", alpha=0.7, label="Q1")
    ax.axhline(0.50, color="#D94F5C", lw=1.5, linestyle="--",             label="Median")
    ax.axhline(0.75, color="#E07B39", lw=1.2, linestyle="--", alpha=0.7, label="Q3")
    ax.set_xlabel("Flash Point (deg C)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Empirical Cumulative Distribution Function")
    ax.legend(fontsize=9)

    plt.tight_layout()
    save_fig("02_target_distribution.png")
    log.info("   Distribution plots generated.")


# ======================================================================
# 4. FUNCTIONAL GROUP ANALYSIS
# ======================================================================

def functional_group_analysis(df: pd.DataFrame):
    """
    Analyses the composition of the dataset by primary functional group:
        - Count bar chart
        - Proportion pie chart
        - Boxplot of flash point by functional group
        - Violin plot of flash point by functional group
    Exports a summary statistics table per group.
    """
    section_header("SECTION 3 -- FUNCTIONAL GROUP ANALYSIS")

    fg_counts = df[GROUP_COL].value_counts()
    fg_order  = fg_counts.index.tolist()
    colors    = [FG_PALETTE.get(g, "#888888") for g in fg_order]

    log.info("Functional group distribution:\n%s", fg_counts.to_string())

    # --- 4a. Count bar chart ---
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(fg_order, fg_counts.values, color=colors, edgecolor="white", linewidth=0.6)
    for bar, count in zip(bars, fg_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(count), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xlabel("Primary Functional Group")
    ax.set_ylabel("Number of Compounds")
    ax.set_title("Dataset Composition by Primary Functional Group", fontweight="bold")
    ax.set_xticklabels(fg_order, rotation=30, ha="right")
    plt.tight_layout()
    save_fig("03a_fg_count_barplot.png")

    # --- 4b. Proportion pie chart ---
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        fg_counts.values,
        labels=fg_order,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.80,
        wedgeprops=dict(linewidth=0.7, edgecolor="white")
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title(
        "Proportion by Primary Functional Group\n(N = {:,} compounds)".format(len(df)),
        fontweight="bold", fontsize=13
    )
    plt.tight_layout()
    save_fig("03b_fg_proportion_pie.png")

    # --- 4c. Boxplot: flash point by functional group ---
    fig, ax = plt.subplots(figsize=(13, 6))
    plot_data = [df.loc[df[GROUP_COL] == g, TARGET_COL].dropna().values for g in fg_order]
    bp = ax.boxplot(
        plot_data, patch_artist=True, notch=False,
        medianprops=dict(color="white", linewidth=2.2),
        whiskerprops=dict(linewidth=1.3),
        capprops=dict(linewidth=1.3),
        flierprops=dict(marker="o", markersize=2.5, alpha=0.35)
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.80)
    ax.set_xticks(range(1, len(fg_order) + 1))
    ax.set_xticklabels(fg_order, rotation=30, ha="right")
    ax.set_xlabel("Primary Functional Group")
    ax.set_ylabel("Flash Point (deg C)")
    ax.set_title("Flash Point Distribution by Functional Group -- Boxplot",
                 fontweight="bold")
    plt.tight_layout()
    save_fig("03c_fg_boxplot.png")

    # --- 4d. Violin plot: flash point by functional group ---
    fig, ax = plt.subplots(figsize=(13, 6))
    vp = ax.violinplot(
        plot_data,
        positions=range(1, len(fg_order) + 1),
        showmedians=True,
        showextrema=True
    )
    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.75)
    vp["cmedians"].set_color("white")
    vp["cmedians"].set_linewidth(2.0)
    ax.set_xticks(range(1, len(fg_order) + 1))
    ax.set_xticklabels(fg_order, rotation=30, ha="right")
    ax.set_xlabel("Primary Functional Group")
    ax.set_ylabel("Flash Point (deg C)")
    ax.set_title("Flash Point Distribution by Functional Group -- Violin Plot",
                 fontweight="bold")
    plt.tight_layout()
    save_fig("03d_fg_violin.png")

    # --- Export per-group summary statistics ---
    group_stats = (
        df.groupby(GROUP_COL)[TARGET_COL]
          .agg(
              N="count",
              Mean="mean",
              Median="median",
              StdDev="std",
              Min="min",
              Max="max",
              Q1=lambda x: x.quantile(0.25),
              Q3=lambda x: x.quantile(0.75),
          )
          .reset_index()
    )
    group_stats.columns = [
        GROUP_COL, "N", "Mean (deg C)", "Median (deg C)", "Std Dev (deg C)",
        "Min (deg C)", "Max (deg C)", "Q1 (deg C)", "Q3 (deg C)"
    ]
    group_stats["IQR (deg C)"] = group_stats["Q3 (deg C)"] - group_stats["Q1 (deg C)"]
    csv_path = os.path.join(OUTPUT_DIR, "03e_fg_summary_statistics.csv")
    group_stats.to_csv(csv_path, index=False)
    log.info("   Summary statistics per functional group saved: %s", csv_path)
    log.info("\n%s", group_stats.to_string(index=False))

    return fg_order, fg_counts


# ======================================================================
# 5. OUTLIER ASSESSMENT
# ======================================================================

def outlier_assessment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects potential outliers in the flash point variable using:
        - IQR-based fences (Tukey method, 1.5x and 3x)
        - Z-score method (|z| > 3 and |z| > 4)
    Produces:
        - Annotated boxplot with fence lines
        - Z-score scatter plot
        - Exports a CSV of detected outlier compounds
    """
    section_header("SECTION 4 -- OUTLIER ASSESSMENT")

    fp = df[TARGET_COL]
    q1  = fp.quantile(0.25)
    q3  = fp.quantile(0.75)
    iqr = q3 - q1

    lower_fence   = q1 - 1.5 * iqr
    upper_fence   = q3 + 1.5 * iqr
    extreme_lower = q1 - 3.0 * iqr
    extreme_upper = q3 + 3.0 * iqr

    iqr_outlier_mask = (fp < lower_fence)  | (fp > upper_fence)
    iqr_extreme_mask = (fp < extreme_lower) | (fp > extreme_upper)

    z_scores      = np.abs(stats.zscore(fp.dropna()))
    z_outlier_idx = fp.dropna().index[z_scores > 3]
    z_extreme_idx = fp.dropna().index[z_scores > 4]

    log.info("IQR Fences   : Lower = %.2f deg C | Upper = %.2f deg C",
             lower_fence, upper_fence)
    log.info("IQR Outliers (1.5xIQR) : %d compounds", iqr_outlier_mask.sum())
    log.info("IQR Extreme  (3.0xIQR) : %d compounds", iqr_extreme_mask.sum())
    log.info("Z-score Outliers (|z|>3): %d compounds", len(z_outlier_idx))
    log.info("Z-score Extreme  (|z|>4): %d compounds", len(z_extreme_idx))

    # --- 5a. Annotated boxplot ---
    fig, ax = plt.subplots(figsize=(5, 9))
    bp = ax.boxplot(
        fp.dropna(), vert=True, patch_artist=True,
        medianprops=dict(color="white", linewidth=2.5),
        whiskerprops=dict(linewidth=1.5, linestyle="--"),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=3, color="#E07B39", alpha=0.40)
    )
    bp["boxes"][0].set_facecolor("#4C8EDA")
    bp["boxes"][0].set_alpha(0.75)
    ax.axhline(lower_fence,   color="#D94F5C", lw=1.5, linestyle="--",
               label=f"Lower fence (1.5xIQR): {lower_fence:.1f} C")
    ax.axhline(upper_fence,   color="#D94F5C", lw=1.5, linestyle="--",
               label=f"Upper fence (1.5xIQR): {upper_fence:.1f} C")
    ax.axhline(extreme_lower, color="#9B59B6", lw=1.2, linestyle=":",
               label=f"Extreme lower (3.0xIQR): {extreme_lower:.1f} C")
    ax.axhline(extreme_upper, color="#9B59B6", lw=1.2, linestyle=":",
               label=f"Extreme upper (3.0xIQR): {extreme_upper:.1f} C")
    ax.set_ylabel("Flash Point (deg C)")
    ax.set_title("Flash Point Boxplot\nwith IQR Outlier Fences", fontweight="bold")
    ax.set_xticks([])
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    save_fig("04a_outlier_boxplot.png")

    # --- 5b. Z-score scatter plot ---
    z_full = pd.Series(
        stats.zscore(fp.fillna(fp.median())),
        index=df.index,
        name="z_score"
    )
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(range(len(z_full)), z_full, s=5, color="#4C8EDA",
               alpha=0.30, label="All compounds")
    mask_out = z_full.abs() > 3
    ax.scatter(np.where(mask_out)[0], z_full[mask_out],
               s=18, color="#E07B39", alpha=0.80, zorder=3, label="|z| > 3")
    mask_ext = z_full.abs() > 4
    ax.scatter(np.where(mask_ext)[0], z_full[mask_ext],
               s=28, color="#D94F5C", zorder=4, label="|z| > 4")
    for threshold, color, style in [(3, "#E07B39", "--"), (-3, "#E07B39", "--"),
                                     (4, "#D94F5C", ":"),  (-4, "#D94F5C", ":")]:
        ax.axhline(threshold, color=color, lw=1.3, linestyle=style)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Compound Index")
    ax.set_ylabel("Z-Score (Flash Point)")
    ax.set_title("Z-Score Analysis -- Flash Point Outlier Detection", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_fig("04b_outlier_zscore.png")

    # --- Export outlier table ---
    outlier_df = df[iqr_outlier_mask].copy()
    outlier_df["z_score"]        = z_full[iqr_outlier_mask].round(3)
    outlier_df["iqr_extreme"]    = iqr_extreme_mask[iqr_outlier_mask].values
    outlier_df["zscore_extreme"] = outlier_df.index.isin(z_extreme_idx)
    csv_path = os.path.join(OUTPUT_DIR, "04c_outlier_compounds.csv")
    outlier_df[["name", "smiles", TARGET_COL, GROUP_COL, "z_score",
                "iqr_extreme", "zscore_extreme"]].to_csv(csv_path, index=False)
    log.info("   Outlier compound list saved: %s", csv_path)

    return outlier_df


# ======================================================================
# 6. CORRELATION ANALYSIS (RDKit Descriptors)
# ======================================================================

def correlation_analysis(df: pd.DataFrame):
    """
    Analyses Pearson correlations between the target variable (flash point)
    and the continuous RDKit physicochemical descriptors.

    Morgan Fingerprints are excluded because:
      * 1024 binary bits are computationally expensive to correlate.
      * Pearson correlation is poorly suited to binary/sparse vectors.
      * Better alternatives (PCA, t-SNE, point-biserial) are recommended
        in the 'Suggested Next Analyses' section.

    Requires: data/processed/featurized_dataset.csv
    """
    section_header("SECTION 5 -- CORRELATION ANALYSIS (RDKit Descriptors)")

    featurized_path = "data/processed/featurized_dataset.csv"
    if not os.path.exists(featurized_path):
        log.warning(
            "Featurized dataset not found at '%s'.\n"
            "Run 'src/feature_engineering.py' to generate it.\n"
            "Correlation analysis will be skipped.", featurized_path
        )
        return None

    log.info("Loading featurized dataset for correlation analysis...")
    df_feat = pd.read_csv(featurized_path)

    available_desc = [c for c in DESCRIPTOR_COLS if c in df_feat.columns]
    if not available_desc:
        log.warning("No RDKit descriptor columns found. Skipping correlation analysis.")
        return None

    numeric_cols = available_desc + [TARGET_COL]
    df_num = df_feat[numeric_cols].apply(pd.to_numeric, errors="coerce").dropna()

    corr_matrix = df_num.corr(method="pearson")
    target_corr = (
        corr_matrix[TARGET_COL]
        .drop(TARGET_COL)
        .sort_values(key=abs, ascending=False)
    )

    log.info("Pearson correlations with Flash Point (sorted by |r|):\n%s",
             target_corr.to_string())

    # --- 6a. Full correlation heatmap (lower triangle) ---
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", mask=mask,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 9}, ax=ax
    )
    ax.set_title(
        "Pearson Correlation Matrix -- RDKit Descriptors + Flash Point",
        fontweight="bold", fontsize=12
    )
    plt.tight_layout()
    save_fig("05a_correlation_heatmap.png")

    # --- 6b. Bar chart: descriptor vs. flash point correlation ---
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = ["#4C8EDA" if v >= 0 else "#D94F5C" for v in target_corr.values]
    bars = ax.bar(target_corr.index, target_corr.values, color=bar_colors,
                  edgecolor="white")
    for bar, val in zip(bars, target_corr.values):
        ypos = bar.get_height() + 0.01 if val >= 0 else bar.get_height() - 0.04
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")
    ax.axhline(0, color="gray", lw=1.0)
    ax.set_xlabel("RDKit Descriptor")
    ax.set_ylabel("Pearson r (vs. Flash Point)")
    ax.set_title("Descriptor Correlations with Flash Point", fontweight="bold")
    ax.set_xticklabels(target_corr.index, rotation=30, ha="right")
    plt.tight_layout()
    save_fig("05b_descriptor_correlation_bar.png")

    # --- 6c. Clustermap (hierarchical clustering of correlations) ---
    clustermap = sns.clustermap(
        corr_matrix, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.4, linecolor="white",
        annot_kws={"size": 8},
        figsize=(12, 10)
    )
    clustermap.fig.suptitle(
        "Hierarchically Clustered Correlation Matrix\n(RDKit Descriptors + Flash Point)",
        fontweight="bold", y=1.01, fontsize=12
    )
    path = os.path.join(OUTPUT_DIR, "05c_correlation_clustermap.png")
    clustermap.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    log.info("   Figure saved: %s", path)

    # Export correlation with target
    csv_path = os.path.join(OUTPUT_DIR, "05d_descriptor_correlations.csv")
    target_corr.to_frame(name="pearson_r").to_csv(csv_path)
    log.info("   Correlation table saved: %s", csv_path)

    return corr_matrix, target_corr


# ======================================================================
# 7. SCATTER PLOTS -- Descriptors vs. Flash Point
# ======================================================================

def scatter_plots(df: pd.DataFrame):
    """
    Generates bivariate scatter plots of each RDKit descriptor against
    the flash point, color-coded by primary functional group.
    A combined multi-panel figure is produced, plus a pairplot for the
    top-4 descriptors correlated with flash point.
    Requires: data/processed/featurized_dataset.csv
    """
    section_header("SECTION 6 -- SCATTER PLOTS (Descriptors vs. Flash Point)")

    featurized_path = "data/processed/featurized_dataset.csv"
    if not os.path.exists(featurized_path):
        log.warning(
            "Featurized dataset not found. Scatter plots will be skipped.\n"
            "Run 'src/feature_engineering.py' first."
        )
        return

    df_feat = pd.read_csv(featurized_path)
    available_desc = [c for c in DESCRIPTOR_COLS if c in df_feat.columns]
    if not available_desc:
        log.warning("No RDKit descriptor columns found. Skipping scatter plots.")
        return

    # Merge functional group info if not already present
    if GROUP_COL not in df_feat.columns and GROUP_COL in df.columns:
        df_feat = df_feat.merge(df[["smiles", GROUP_COL]], on="smiles", how="left")

    has_group_col = GROUP_COL in df_feat.columns

    # --- 7a. Multi-panel scatter grid (all descriptors) ---
    n_cols = 3
    n_rows = int(np.ceil(len(available_desc) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4.5))
    axes = axes.flatten()

    for i, desc in enumerate(available_desc):
        ax = axes[i]
        if has_group_col:
            for group, gdf in df_feat.groupby(GROUP_COL):
                color = FG_PALETTE.get(group, "#888888")
                ax.scatter(gdf[desc], gdf[TARGET_COL],
                           s=8, alpha=0.30, color=color, label=group)
        else:
            ax.scatter(df_feat[desc], df_feat[TARGET_COL],
                       s=8, alpha=0.30, color="#4C8EDA")

        # OLS regression line
        clean = df_feat[[desc, TARGET_COL]].dropna()
        if len(clean) > 2:
            m, b, r, p, _ = stats.linregress(clean[desc], clean[TARGET_COL])
            x_line = np.linspace(clean[desc].min(), clean[desc].max(), 200)
            ax.plot(x_line, m * x_line + b, color="#E07B39", lw=1.8,
                    label=f"r = {r:.3f}")
        ax.set_xlabel(desc)
        ax.set_ylabel("Flash Point (deg C)")
        ax.set_title(f"{desc} vs. Flash Point", fontsize=10)
        ax.legend(fontsize=6, ncol=2, loc="upper left")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "RDKit Descriptor Scatter Plots vs. Flash Point\n(colored by Functional Group)",
        fontweight="bold", fontsize=14, y=1.01
    )
    plt.tight_layout()
    save_fig("06a_scatter_all_descriptors.png")

    # --- 7b. Pairplot of top-4 descriptors + flash point ---
    top4 = [c for c in ["MW", "HeavyAtoms", "MolarRefractivity", "LogP"]
            if c in df_feat.columns]
    pair_cols = top4 + [TARGET_COL]
    df_pair = df_feat[pair_cols + ([GROUP_COL] if has_group_col else [])].dropna()

    if has_group_col:
        unique_groups = df_pair[GROUP_COL].unique()
        palette_filtered = {g: FG_PALETTE.get(g, "#888888") for g in unique_groups}
        pp = sns.pairplot(
            df_pair, hue=GROUP_COL, palette=palette_filtered,
            vars=pair_cols, plot_kws={"s": 8, "alpha": 0.30},
            diag_kind="kde"
        )
    else:
        pp = sns.pairplot(
            df_pair, vars=pair_cols,
            plot_kws={"s": 8, "alpha": 0.30}, diag_kind="kde"
        )

    pp.fig.suptitle("Pairplot -- Top Descriptors + Flash Point",
                    fontweight="bold", fontsize=13, y=1.01)
    path = os.path.join(OUTPUT_DIR, "06b_pairplot_top_descriptors.png")
    pp.savefig(path, bbox_inches="tight", dpi=180)
    plt.close()
    log.info("   Figure saved: %s", path)


# ======================================================================
# 8. FEATURE DISTRIBUTION HISTOGRAMS (RDKit Descriptors)
# ======================================================================

def descriptor_histograms(df: pd.DataFrame):
    """
    Plots histograms for each RDKit physicochemical descriptor, overlaying
    distributions for the three largest functional groups to highlight
    group-specific structural patterns.
    Requires: data/processed/featurized_dataset.csv
    """
    section_header("SECTION 7 -- RDKit DESCRIPTOR HISTOGRAMS")

    featurized_path = "data/processed/featurized_dataset.csv"
    if not os.path.exists(featurized_path):
        log.warning("Featurized dataset not found. Histograms will be skipped.")
        return

    df_feat = pd.read_csv(featurized_path)
    available_desc = [c for c in DESCRIPTOR_COLS if c in df_feat.columns]
    if not available_desc:
        log.warning("No RDKit descriptor columns found. Skipping histograms.")
        return

    if GROUP_COL not in df_feat.columns and GROUP_COL in df.columns:
        df_feat = df_feat.merge(df[["smiles", GROUP_COL]], on="smiles", how="left")

    top3_groups = (
        df[GROUP_COL].value_counts().head(3).index.tolist()
        if GROUP_COL in df.columns else []
    )

    n_cols = 2
    n_rows = int(np.ceil(len(available_desc) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))
    axes = axes.flatten()

    for i, desc in enumerate(available_desc):
        ax = axes[i]
        data_all = df_feat[desc].dropna()
        ax.hist(data_all, bins=40, color="#4C8EDA", alpha=0.35,
                edgecolor="white", density=True, label="All compounds")
        if GROUP_COL in df_feat.columns:
            for group in top3_groups:
                gdata = df_feat.loc[df_feat[GROUP_COL] == group, desc].dropna()
                if len(gdata) > 5:
                    ax.hist(gdata, bins=30, alpha=0.55, density=True,
                            edgecolor="white",
                            color=FG_PALETTE.get(group, "#888"),
                            label=group)
        ax.set_xlabel(desc)
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution of {desc}", fontsize=10)
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "RDKit Descriptor Distributions (All Compounds + Top-3 Functional Groups)",
        fontweight="bold", fontsize=13, y=1.01
    )
    plt.tight_layout()
    save_fig("07_descriptor_histograms.png")
    log.info("   Descriptor histograms generated.")


# ======================================================================
# 9. SUGGESTED NEXT ANALYSES
# ======================================================================

SUGGESTED_ANALYSES = """
================================================================================
  SUGGESTED NEXT ANALYSES -- ChemPyQSPR Flash Point EDA
================================================================================

The following analyses are recommended to extend this EDA and further improve
model performance. They are ordered from highest to lowest expected impact.

--------------------------------------------------------------------------------
A. MORGAN FINGERPRINT DIMENSIONALITY REDUCTION
--------------------------------------------------------------------------------
  Rationale: Morgan fingerprints (1024 binary bits) cannot be effectively
  explored with standard correlation metrics due to sparsity and binary nature.

  Recommended approaches:
    1. PCA on fingerprint matrix  -> visualize chemical space in 2D/3D.
    2. t-SNE or UMAP              -> non-linear embedding for cluster discovery.
    3. Point-biserial correlation -> identify individual bits most correlated
       with flash point.
    4. Bit frequency analysis     -> remove near-zero-variance bits (>95%
       constant) before modeling to reduce noise.

--------------------------------------------------------------------------------
B. STATISTICAL TESTS: GROUP DIFFERENCES
--------------------------------------------------------------------------------
  Rationale: Box/violin plots suggest flash point distributions differ
  significantly across functional groups.

  Recommended analyses:
    1. Kruskal-Wallis H test              -> non-parametric one-way ANOVA.
    2. Dunn post-hoc test (Bonferroni)    -> pairwise group comparison.
    3. Effect size (eta^2, Cohen's d)     -> quantify practical significance.

--------------------------------------------------------------------------------
C. POLYFUNCTIONAL COMPOUND ANALYSIS
--------------------------------------------------------------------------------
  Rationale: The 'all_functional_groups' column encodes multi-group compounds
  (e.g., amino acids, hydroxy-esters). These may follow different flash point
  trends than single-group compounds.

  Recommended analyses:
    1. Parse 'all_functional_groups' -> count co-occurring group combinations.
    2. Compare flash point: single-group vs. multi-group compounds.
    3. Build a co-occurrence heatmap across all group pairs.
    4. Assess whether polyfunctional compounds cluster in model residuals
       (high prediction error regions).

--------------------------------------------------------------------------------
D. FLASH POINT vs. BOILING POINT / MOLECULAR WEIGHT REGRESSION
--------------------------------------------------------------------------------
  Rationale: Several empirical correlations (Catoire, Butler-Walsh) link flash
  point to boiling point or MW. Quantifying this baseline reveals how much
  additional information fingerprints/descriptors add beyond simple proxies.

  Recommended analyses:
    1. Segment-wise OLS: FP ~ MW per functional group.
    2. Residual analysis: does a simple MW model leave systematic bias for
       specific functional groups?

--------------------------------------------------------------------------------
E. CARBON CHAIN LENGTH / DEGREE OF UNSATURATION
--------------------------------------------------------------------------------
  Rationale: For hydrocarbons and esters, chain length and degree of
  unsaturation are primary determinants of flash point.

  Recommended analyses:
    1. Extract number of carbons and degree of unsaturation from SMILES.
    2. Plot FP vs. chain length within each functional group.
    3. Evaluate whether these derived features improve model R2.

--------------------------------------------------------------------------------
F. FEATURE IMPORTANCE FROM TREE-BASED MODELS
--------------------------------------------------------------------------------
  Rationale: Random Forest / XGBoost trained in train_model.py provide feature
  importance scores that complement Pearson correlations, especially for
  non-linear relationships.

  Recommended analyses:
    1. Extract and plot MDI feature importances from Random Forest.
    2. Run SHAP (SHapley Additive exPlanations) for model-agnostic importance.
    3. Permutation importance for unbiased estimation with correlated features.

--------------------------------------------------------------------------------
G. OUTLIER INVESTIGATION
--------------------------------------------------------------------------------
  Rationale: The outlier CSV (04c_outlier_compounds.csv) lists compounds with
  extreme flash points. Some may be valid (peroxides, explosives) while others
  may be data entry errors.

  Recommended analyses:
    1. Cross-reference outliers with PubChem / NIST WebBook for confirmation.
    2. Check whether outliers cluster in specific functional groups.
    3. Train models with and without outliers -> compare test-set RMSE.

--------------------------------------------------------------------------------
H. TRAIN / TEST CHEMICAL SPACE OVERLAP
--------------------------------------------------------------------------------
  Rationale: The 80/20 stratified split may still produce sets with limited
  chemical diversity overlap, affecting model generalizability.

  Recommended analyses:
    1. Tanimoto similarity distribution between train and test fingerprints.
    2. PCA/t-SNE visualization of train vs. test chemical space.
    3. Application Domain analysis (distance-to-model metric).

================================================================================
"""


def print_suggested_analyses():
    """Prints the suggested next analyses to the terminal and saves to file."""
    section_header("SECTION 8 -- SUGGESTED NEXT ANALYSES")
    print(SUGGESTED_ANALYSES)

    path = os.path.join(OUTPUT_DIR, "08_suggested_next_analyses.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(SUGGESTED_ANALYSES)
    log.info("   Suggested analyses saved: %s", path)


# ======================================================================
# MAIN PIPELINE
# ======================================================================

def run_eda_pipeline():
    """
    Orchestrates all EDA sections in sequence.
    """
    log.info("=" * 75)
    log.info("  ChemPyQSPR -- Exploratory Data Analysis (EDA) Pipeline")
    log.info("=" * 75)
    log.info("  Input   : %s", INPUT_CSV)
    log.info("  Output  : %s/", OUTPUT_DIR)
    log.info("=" * 75)

    # 1. Load data
    df = load_dataset(INPUT_CSV)
    log.info("Dataset shape: %s | Functional groups: %d",
             df.shape, df[GROUP_COL].nunique())

    # 2. Descriptive statistics
    descriptive_statistics(df)

    # 3. Target distribution
    plot_target_distribution(df)

    # 4. Functional group analysis
    functional_group_analysis(df)

    # 5. Outlier assessment
    outlier_assessment(df)

    # 6. Correlation analysis (requires featurized_dataset.csv)
    correlation_analysis(df)

    # 7. Scatter plots (requires featurized_dataset.csv)
    scatter_plots(df)

    # 8. Descriptor histograms (requires featurized_dataset.csv)
    descriptor_histograms(df)

    # 9. Suggested next analyses
    print_suggested_analyses()

    log.info("")
    log.info("=" * 75)
    log.info("  EDA PIPELINE COMPLETED SUCCESSFULLY!")
    log.info("  All outputs saved to: %s/", OUTPUT_DIR)
    log.info("=" * 75)


if __name__ == "__main__":
    run_eda_pipeline()