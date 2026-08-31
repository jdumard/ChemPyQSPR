"""
statistical_tests.py
====================
Statistical tests for functional group differences in the ChemPyQSPR project.

This script performs non-parametric statistical tests to determine if the 
flash point distributions of different functional groups are significantly different.
It performs:
    1. Kruskal-Wallis H Test (Global non-parametric ANOVA)
    2. Pairwise Mann-Whitney U Tests with Bonferroni Correction (Post-hoc)
    3. Effect Size Calculation (Epsilon-squared)

Outputs:
    Figures and CSVs are saved to: data/processed/statistical_results/

Usage:
    Run from the project root:
        python src/statistical_tests.py
"""

import os
import sys
import logging
import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

INPUT_CSV = "data/processed/cleaned_flash_point_database_fg.csv"
OUTPUT_DIR = "data/processed/statistical_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL = "flash_point"
GROUP_COL = "primary_functional_group"

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

def save_fig(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path)
    plt.close()
    log.info("   Figure saved: %s", path)

def run_statistical_tests():
    log.info("=" * 75)
    log.info("  ChemPyQSPR -- Statistical Tests (Functional Groups)")
    log.info("=" * 75)

    if not os.path.exists(INPUT_CSV):
        log.error("Input file not found: %s", INPUT_CSV)
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    df = df.dropna(subset=[TARGET_COL, GROUP_COL])
    
    log.info("Dataset loaded: N = %d valid compounds", len(df))
    
    groups = df[GROUP_COL].unique()
    log.info("Number of functional groups: %d", len(groups))
    
    # -------------------------------------------------------------------
    # 1. Kruskal-Wallis H Test
    # -------------------------------------------------------------------
    log.info("-" * 75)
    log.info("1. Kruskal-Wallis H Test (Global)")
    
    # Prepare list of arrays for the test
    group_data = [df[df[GROUP_COL] == g][TARGET_COL].values for g in groups]
    
    h_stat, p_val = kruskal(*group_data)
    
    N = len(df)
    # Epsilon-squared effect size for Kruskal-Wallis: E^2 = H / ((N^2 - 1) / (N + 1)) = H / (N - 1)
    epsilon_sq = h_stat / (N - 1)
    
    log.info("Kruskal-Wallis H-statistic : %.2f", h_stat)
    log.info("Kruskal-Wallis p-value     : %.2e", p_val)
    log.info("Effect size (Epsilon^2)    : %.4f", epsilon_sq)
    
    if p_val < 0.05:
        log.info("Conclusion: There is a statistically significant difference between the groups.")
    else:
        log.info("Conclusion: No significant difference between the groups.")
        
    global_results = pd.DataFrame({
        "Test": ["Kruskal-Wallis"],
        "H_statistic": [h_stat],
        "p_value": [p_val],
        "Effect_size_epsilon_sq": [epsilon_sq]
    })
    global_results.to_csv(os.path.join(OUTPUT_DIR, "01_kruskal_wallis_global.csv"), index=False)
    
    # -------------------------------------------------------------------
    # 2. Pairwise Mann-Whitney U with Bonferroni Correction
    # -------------------------------------------------------------------
    log.info("-" * 75)
    log.info("2. Pairwise Post-hoc Tests (Mann-Whitney U with Bonferroni)")
    
    pairs = list(itertools.combinations(groups, 2))
    num_comparisons = len(pairs)
    log.info("Total pairwise comparisons: %d", num_comparisons)
    
    results = []
    
    for g1, g2 in pairs:
        data1 = df[df[GROUP_COL] == g1][TARGET_COL].values
        data2 = df[df[GROUP_COL] == g2][TARGET_COL].values
        
        # two-sided test
        stat, p = mannwhitneyu(data1, data2, alternative='two-sided')
        
        # Bonferroni correction
        p_adj = min(p * num_comparisons, 1.0)
        
        results.append({
            "Group_1": g1,
            "Group_2": g2,
            "U_statistic": stat,
            "p_value_raw": p,
            "p_value_adj": p_adj,
            "Significant": p_adj < 0.05
        })
        
    pairwise_df = pd.DataFrame(results)
    pairwise_df = pairwise_df.sort_values(by="p_value_adj").reset_index(drop=True)
    pairwise_df.to_csv(os.path.join(OUTPUT_DIR, "02_pairwise_mann_whitney.csv"), index=False)
    
    num_significant = pairwise_df["Significant"].sum()
    log.info("Significant pairwise differences (p_adj < 0.05): %d out of %d (%.1f%%)", 
             num_significant, num_comparisons, (num_significant/num_comparisons)*100)
    
    # -------------------------------------------------------------------
    # 3. Heatmap of Adjusted P-values
    # -------------------------------------------------------------------
    log.info("-" * 75)
    log.info("3. Generating P-value Heatmap")
    
    # Create an empty matrix
    sorted_groups = sorted(list(groups))
    p_matrix = pd.DataFrame(1.0, index=sorted_groups, columns=sorted_groups)
    
    for _, row in pairwise_df.iterrows():
        g1, g2, p_adj = row['Group_1'], row['Group_2'], row['p_value_adj']
        p_matrix.loc[g1, g2] = p_adj
        p_matrix.loc[g2, g1] = p_adj
        
    # Apply -log10 transformation for better visualization of small p-values
    # Replace 0 with a very small number (e.g., 1e-300) to avoid log(0)
    p_matrix_val = p_matrix.replace(0, 1e-300)
    log_p_matrix = -np.log10(p_matrix_val)
    
    plt.figure(figsize=(11, 9))
    mask = np.triu(np.ones_like(log_p_matrix, dtype=bool))
    
    # Define a custom colormap where non-significant (< 1.3, which is -log10(0.05)) is grey
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    cmap.set_under("lightgrey")
    
    sns.heatmap(log_p_matrix, mask=mask, cmap=cmap, vmin=1.301, 
                annot=False, cbar_kws={'label': '-log10(Adjusted p-value)'})
    
    plt.title("Significance of Pairwise Differences (-log10 p-value)\nGrey = Not Significant (p >= 0.05)", fontweight="bold")
    plt.xlabel("")
    plt.ylabel("")
    save_fig("03_pairwise_significance_heatmap.png")

    log.info("=" * 75)
    log.info("  STATISTICAL TESTS COMPLETED SUCCESSFULLY!")
    log.info("=" * 75)

if __name__ == "__main__":
    run_statistical_tests()