"""
polyfunctional_analysis.py
==========================
Polyfunctional Compound Analysis for the ChemPyQSPR Project.

This script analyzes compounds containing multiple functional groups to determine:
    1. Prevalence of polyfunctional vs. single-functional compounds.
    2. Most common co-occurring functional groups (co-occurrence heatmap).
    3. Impact of polyfunctionality on Flash Point distributions.

Outputs:
    All figures are saved to: data/processed/polyfunctional_results/
    Summary tables are exported to CSV in the same folder.

Usage:
    Run from the project root:
        python src/polyfunctional_analysis.py
"""

import os
import sys
import logging
import warnings
import ast
import itertools
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

INPUT_CSV = "data/processed/cleaned_flash_point_database_fg.csv"
OUTPUT_DIR = "data/processed/polyfunctional_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL = "flash_point"
ALL_FG_COL = "all_functional_groups"

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})

def parse_groups(val):
    """Safely parse the string representation of a list into an actual list."""
    if pd.isna(val):
        return []
    if isinstance(val, str):
        if ";" in val:
            return [x.strip() for x in val.split(";")]
        try:
            res = ast.literal_eval(val)
            if isinstance(res, list): return res
        except:
            pass
        if "[" in val:
            return [x.strip(" '\"[]") for x in val.split(',')]
        return [val.strip()]
    elif isinstance(val, list):
        return val
    return []

def save_fig(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path)
    plt.close()
    log.info("   Figure saved: %s", path)

def run_polyfunctional_analysis():
    log.info("=" * 75)
    log.info("  ChemPyQSPR -- Polyfunctional Compound Analysis")
    log.info("=" * 75)

    if not os.path.exists(INPUT_CSV):
        log.error("Input file not found: %s", INPUT_CSV)
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    df = df.dropna(subset=[TARGET_COL])
    
    log.info("Dataset loaded: N = %d valid compounds", len(df))

    # 1. Parse Groups and Count
    log.info("Parsing 'all_functional_groups' column...")
    df['parsed_groups'] = df[ALL_FG_COL].apply(parse_groups)
    
    # Remove 'hydrocarbon' from parsed_groups if there are other functional groups
    # because every molecule has a hydrocarbon backbone, but we are interested in true polyfunctionality
    def filter_hydrocarbon(groups):
        if len(groups) > 1 and "hydrocarbon" in groups:
            return [g for g in groups if g != "hydrocarbon"]
        return groups
        
    df['parsed_groups'] = df['parsed_groups'].apply(filter_hydrocarbon)
    df['group_count'] = df['parsed_groups'].apply(len)
    
    # Categorize single vs poly
    df['functionality_type'] = np.where(df['group_count'] > 1, 'Polyfunctional (>1)', 'Single-functional (1)')
    df.loc[df['group_count'] == 0, 'functionality_type'] = 'Unknown (0)'
    
    counts = df['functionality_type'].value_counts()
    poly_ratio = (counts.get('Polyfunctional (>1)', 0) / len(df)) * 100
    log.info("Composition: Single=%.1f%% | Polyfunctional=%.1f%%", 100 - poly_ratio, poly_ratio)
    
    # --- Plot: Functionality Type Distribution ---
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(data=df, x='functionality_type', palette="Set2", order=['Single-functional (1)', 'Polyfunctional (>1)'])
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.title("Single vs Polyfunctional Compounds", fontweight="bold")
    plt.ylabel("Number of Compounds")
    plt.xlabel("")
    save_fig("01_functionality_distribution.png")
    
    # 2. Flash Point Comparison (Single vs Poly)
    log.info("Comparing Flash Point distributions...")
    
    plt.figure(figsize=(7, 5))
    sns.kdeplot(data=df, x=TARGET_COL, hue='functionality_type', fill=True, common_norm=False, palette="Set2")
    plt.title("Flash Point Distribution: Single vs Polyfunctional", fontweight="bold")
    plt.xlabel("Flash Point (deg C)")
    save_fig("02_flash_point_kde.png")
    
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=df, x='functionality_type', y=TARGET_COL, palette="Set2", order=['Single-functional (1)', 'Polyfunctional (>1)'])
    plt.title("Flash Point by Functionality Type", fontweight="bold")
    plt.ylabel("Flash Point (deg C)")
    plt.xlabel("")
    save_fig("03_flash_point_boxplot.png")
    
    # Statistics
    stats_df = df.groupby('functionality_type')[TARGET_COL].agg(['count', 'mean', 'median', 'std']).round(2)
    stats_df.to_csv(os.path.join(OUTPUT_DIR, "04_functionality_stats.csv"))
    log.info("\n%s\n", stats_df.to_string())

    # 3. Co-occurrence Analysis
    log.info("Analyzing functional group co-occurrences...")
    poly_df = df[df['group_count'] > 1]
    
    if len(poly_df) > 0:
        all_pairs = []
        for groups in poly_df['parsed_groups']:
            # Sort to avoid (A,B) and (B,A) being counted separately
            groups_sorted = sorted(list(set(groups)))
            pairs = list(itertools.combinations(groups_sorted, 2))
            all_pairs.extend(pairs)
            
        pair_counts = Counter(all_pairs)
        
        # Create a DataFrame for the heatmap
        unique_groups = sorted(list(set(itertools.chain.from_iterable(poly_df['parsed_groups']))))
        co_matrix = pd.DataFrame(0, index=unique_groups, columns=unique_groups)
        
        for (g1, g2), count in pair_counts.items():
            co_matrix.loc[g1, g2] = count
            co_matrix.loc[g2, g1] = count # Symmetric matrix
            
        co_matrix.to_csv(os.path.join(OUTPUT_DIR, "05_co_occurrence_matrix.csv"))
        
        # Plot Heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(co_matrix, dtype=bool))
        sns.heatmap(co_matrix, mask=mask, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={"shrink": .8})
        plt.title("Functional Group Co-occurrence in Polyfunctional Compounds", fontweight="bold")
        plt.ylabel("")
        plt.xlabel("")
        save_fig("06_co_occurrence_heatmap.png")
        
        # Top 10 pairs
        top_pairs = pair_counts.most_common(10)
        log.info("Top 5 Co-occurring Pairs:")
        for pair, count in top_pairs[:5]:
            log.info("   %s + %s : %d compounds", pair[0], pair[1], count)
            
        pairs_df = pd.DataFrame([{"Group_1": p[0], "Group_2": p[1], "Count": c} for p, c in top_pairs])
        pairs_df.to_csv(os.path.join(OUTPUT_DIR, "07_top_co_occurring_pairs.csv"), index=False)
    else:
        log.warning("No polyfunctional compounds found to build co-occurrence matrix.")

    log.info("=" * 75)
    log.info("  POLYFUNCTIONAL ANALYSIS COMPLETED SUCCESSFULLY!")
    log.info("=" * 75)

if __name__ == "__main__":
    run_polyfunctional_analysis()