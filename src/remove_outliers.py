"""
remove_outliers.py
==================
Modular Data Curation and Outlier Removal Pipeline for ChemPyQSPR.

Implements a systematic 3-stage chemical and statistical curation workflow:
  • Stage 1: Radical & Fragment Filter (unclosed valences, radical species, invalid SMILES)
  • Stage 2: Physicochemical Consistency Filter (Kelvin/Celsius unit artifacts, MW vs. FP thermodynamic sanity)
  • Stage 3: Statistical Residual Outlier Filter (|z| > 3.0σ from OOF Cross-Validation)

Each stage is completely decoupled and can be enabled or disabled individually
via command-line flags or function arguments.

Outputs:
  - Curated featurized dataset : data/processed/featurized_dataset_curated.csv
  - Curated metadata + FG CSV  : data/processed/cleaned_flash_point_database_curated_fg.csv
  - Full audit log of exclusions: data/processed/curation_results/excluded_outliers_audit.csv
  - Summary metrics JSON       : data/processed/curation_results/curation_summary.json
  - Visual before/after plots  : data/processed/curation_results/*.png

Usage:
  # Run all 3 stages (recommended):
  python src/remove_outliers.py

  # Run only Stage 1 (Radicals) and Stage 2 (Physicochemical):
  python src/remove_outliers.py --skip-statistical

  # Run only Stage 3 (Statistical Residual Outliers):
  python src/remove_outliers.py --skip-radicals --skip-physchem
"""

# ----------------------------------------------------------------------
# Imports & Configuration
# ----------------------------------------------------------------------
import os
import sys
import json
import logging
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import Descriptors

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
INPUT_FEATURIZED_CSV = "data/processed/featurized_dataset.csv"
INPUT_FG_CSV         = "data/processed/cleaned_flash_point_database_fg.csv"
RESIDUALS_CSV        = "data/processed/residual_analysis/cv_predictions_and_residuals.csv"

OUTPUT_DIR           = "data/processed/curation_results"
OUTPUT_FEATURIZED_CSV = "data/processed/featurized_dataset_curated.csv"
OUTPUT_FG_CSV         = "data/processed/cleaned_flash_point_database_curated_fg.csv"
AUDIT_LOG_CSV         = os.path.join(OUTPUT_DIR, "excluded_outliers_audit.csv")
SUMMARY_JSON          = os.path.join(OUTPUT_DIR, "curation_summary.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Seaborn & Matplotlib Styles
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight"
})


# ======================================================================
# Section 1 -- Data Ingestion
# ======================================================================

def load_datasets(
    featurized_path: str = INPUT_FEATURIZED_CSV,
    fg_path: str = INPUT_FG_CSV
) -> tuple:
    """
    Loads the featurized matrix and the metadata/functional groups dataset.
    """
    if not os.path.exists(featurized_path):
        log.error("Featurized dataset not found at: %s", featurized_path)
        sys.exit(1)

    log.info("Loading featurized dataset from: %s", featurized_path)
    df_feat = pd.read_csv(featurized_path)

    df_fg = None
    if os.path.exists(fg_path):
        log.info("Loading functional groups dataset from: %s", fg_path)
        df_fg = pd.read_csv(fg_path)

    log.info("Initial dataset loaded: N = %d compounds.", len(df_feat))
    return df_feat, df_fg


# ======================================================================
# Section 2 -- Stage 1: Radical & Fragment Species Filter
# ======================================================================

def filter_radical_and_fragment_species(
    df: pd.DataFrame,
    smiles_col: str = "smiles"
) -> tuple:
    """
    Identifies and removes unclosed-valence radicals, open-shell reactive
    intermediates, and unparseable SMILES structures.

    Parameters
    ----------
    df         : pd.DataFrame -- Input chemical dataset
    smiles_col : str          -- Column containing SMILES strings

    Returns
    -------
    df_clean   : pd.DataFrame -- Cleaned subset
    df_dropped : pd.DataFrame -- Dropped records with exclusion metadata
    """
    log.info("-" * 75)
    log.info("STAGE 1: Evaluating Radicals, Unclosed Valences & Fragment Species...")
    log.info("-" * 75)

    dropped_records = []
    keep_mask = []

    for idx, row in df.iterrows():
        smiles = row.get(smiles_col)
        name   = row.get("name", f"Record_{idx}")
        fp_val = row.get("flash_point", np.nan)
        fg_val = row.get("primary_functional_group", "unknown")

        if pd.isna(smiles) or not isinstance(smiles, str) or not smiles.strip():
            dropped_records.append({
                "name": name,
                "smiles": smiles,
                "flash_point": fp_val,
                "primary_functional_group": fg_val,
                "stage": "Stage_1_Radical_Fragment",
                "exclusion_reason": "Missing or empty SMILES string"
            })
            keep_mask.append(False)
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            dropped_records.append({
                "name": name,
                "smiles": smiles,
                "flash_point": fp_val,
                "primary_functional_group": fg_val,
                "stage": "Stage_1_Radical_Fragment",
                "exclusion_reason": "Unparseable SMILES structure (RDKit parse failure)"
            })
            keep_mask.append(False)
            continue

        # Check for free radical electrons
        num_radicals = sum([atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()])
        if num_radicals > 0:
            dropped_records.append({
                "name": name,
                "smiles": smiles,
                "flash_point": fp_val,
                "primary_functional_group": fg_val,
                "stage": "Stage_1_Radical_Fragment",
                "exclusion_reason": f"Free radical species (radical electrons = {num_radicals})"
            })
            keep_mask.append(False)
            continue

        keep_mask.append(True)

    df_clean = df[keep_mask].copy()
    df_dropped = pd.DataFrame(dropped_records)

    log.info("Stage 1 complete: %d radical/fragment record(s) identified and flagged.", len(df_dropped))
    return df_clean, df_dropped


# ======================================================================
# Section 3 -- Stage 2: Physicochemical Consistency Filter
# ======================================================================

def filter_physicochemical_inconsistencies(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    target_col: str = "flash_point",
    mw_col: str = "MW"
) -> tuple:
    """
    Applies thermodynamic sanity checks and detects unit conversion errors
    (e.g., Kelvin or Fahrenheit recorded as Celsius in raw databases).

    Rules:
      1. High Molecular Weight with Sub-zero Flash Point:
         If MW > 350 g/mol and Flash Point < 0 °C (physically impossible for heavy organic solids/liquids).
      2. Extremely Light Compounds with High Flash Point:
         If MW < 85 g/mol and Flash Point > 220 °C (indicative of Kelvin unit error, e.g. 194 K = -79 °C recorded as +194 °C).
      3. Heavy Solids with Cryogenic Flash Point:
         If MW > 250 g/mol and Flash Point < -20 °C.
    """
    log.info("-" * 75)
    log.info("STAGE 2: Evaluating Physicochemical & Thermodynamic Consistency...")
    log.info("-" * 75)

    dropped_records = []
    keep_mask = []

    for idx, row in df.iterrows():
        smiles = row.get(smiles_col)
        name   = row.get("name", f"Record_{idx}")
        fp     = float(row.get(target_col, np.nan))
        fg     = row.get("primary_functional_group", "unknown")

        # Get or compute molecular weight
        if mw_col in df.columns and pd.notna(row.get(mw_col)):
            mw = float(row[mw_col])
        else:
            mol = Chem.MolFromSmiles(smiles) if pd.notna(smiles) else None
            mw = Descriptors.MolWt(mol) if mol else np.nan

        if np.isnan(fp) or np.isnan(mw):
            keep_mask.append(True)
            continue

        # Rule 1: Heavy MW with cryogenic flash point
        if mw > 350.0 and fp < 0.0:
            dropped_records.append({
                "name": name,
                "smiles": smiles,
                "flash_point": fp,
                "primary_functional_group": fg,
                "stage": "Stage_2_Physicochemical",
                "exclusion_reason": f"Thermodynamic inconsistency: Heavy molecule (MW={mw:.1f} g/mol) with cryogenic FP ({fp:.1f} °C)"
            })
            keep_mask.append(False)
            continue

        # Rule 2: Ultra-light volatile compound with extreme high flash point (Kelvin artifact)
        if mw < 85.0 and fp > 220.0:
            dropped_records.append({
                "name": name,
                "smiles": smiles,
                "flash_point": fp,
                "primary_functional_group": fg,
                "stage": "Stage_2_Physicochemical",
                "exclusion_reason": f"Unit artifact candidate: Low MW ({mw:.1f} g/mol) with extreme high FP ({fp:.1f} °C > 220 °C; probable Kelvin error)"
            })
            keep_mask.append(False)
            continue

        # Rule 3: Medium-heavy molecule (MW > 250 g/mol) with severe cryogenic FP (< -20 °C)
        if mw > 250.0 and fp < -20.0:
            dropped_records.append({
                "name": name,
                "smiles": smiles,
                "flash_point": fp,
                "primary_functional_group": fg,
                "stage": "Stage_2_Physicochemical",
                "exclusion_reason": f"Thermodynamic inconsistency: MW={mw:.1f} g/mol with severe sub-zero FP ({fp:.1f} °C < -20 °C)"
            })
            keep_mask.append(False)
            continue

        keep_mask.append(True)

    df_clean = df[keep_mask].copy()
    df_dropped = pd.DataFrame(dropped_records)

    log.info("Stage 2 complete: %d physicochemical unit/inconsistency record(s) flagged.", len(df_dropped))
    return df_clean, df_dropped


# ======================================================================
# Section 4 -- Stage 3: Statistical Residual Outlier Filter (|z| > 3.0σ)
# ======================================================================

def filter_statistical_residual_outliers(
    df: pd.DataFrame,
    residuals_csv_path: str = RESIDUALS_CSV,
    z_threshold: float = 3.0,
    smiles_col: str = "smiles"
) -> tuple:
    """
    Filters statistical extreme outliers (|z| > 3.0σ) identified during
    out-of-fold cross-validation.
    """
    log.info("-" * 75)
    log.info("STAGE 3: Evaluating Statistical Residual Outliers (|z| > %.1f sigma)...", z_threshold)
    log.info("-" * 75)

    if not os.path.exists(residuals_csv_path):
        log.warning("Residuals CSV not found at '%s'. Skipping Stage 3.", residuals_csv_path)
        log.warning("Run 'python src/cross_validation_and_residuals.py' first to compute OOF residuals.")
        return df, pd.DataFrame()

    df_res = pd.read_csv(residuals_csv_path)

    # Identify outlier SMILES / identifiers
    if "std_residual" in df_res.columns:
        outlier_mask = np.abs(df_res["std_residual"]) > z_threshold
    elif "is_extreme_outlier" in df_res.columns:
        outlier_mask = df_res["is_extreme_outlier"] == True
    else:
        log.warning("Could not find 'std_residual' column in '%s'. Skipping Stage 3.", residuals_csv_path)
        return df, pd.DataFrame()

    df_outliers_ref = df_res[outlier_mask].copy()
    outlier_smiles_set = set(df_outliers_ref[smiles_col].dropna().unique())

    log.info("Identified %d unique outlier SMILES from cross-validation analysis.", len(outlier_smiles_set))

    dropped_records = []
    keep_mask = []

    for idx, row in df.iterrows():
        smi = row.get(smiles_col)
        name = row.get("name", f"Record_{idx}")
        fp   = row.get("flash_point", np.nan)
        fg   = row.get("primary_functional_group", "unknown")

        if smi in outlier_smiles_set:
            match = df_outliers_ref[df_outliers_ref[smiles_col] == smi]
            z_val = match["std_residual"].values[0] if "std_residual" in match.columns else np.nan
            res_val = match["residual"].values[0] if "residual" in match.columns else np.nan

            dropped_records.append({
                "name": name,
                "smiles": smi,
                "flash_point": fp,
                "primary_functional_group": fg,
                "stage": "Stage_3_Statistical_Residual",
                "exclusion_reason": f"Statistical outlier: Residual={res_val:.1f} °C, Standardized residual z={z_val:.2f}σ (|z| > {z_threshold}σ)"
            })
            keep_mask.append(False)
        else:
            keep_mask.append(True)

    df_clean = df[keep_mask].copy()
    df_dropped = pd.DataFrame(dropped_records)

    log.info("Stage 3 complete: %d statistical residual outlier record(s) flagged.", len(df_dropped))
    return df_clean, df_dropped


# ======================================================================
# Section 5 -- Diagnostic & Comparison Visualizations
# ======================================================================

def generate_curation_visualizations(
    df_original: pd.DataFrame,
    df_curated: pd.DataFrame,
    df_all_dropped: pd.DataFrame,
    output_dir: str = OUTPUT_DIR
) -> None:
    """
    Plots before/after target distributions and functional group retention rates.
    """
    log.info("-" * 75)
    log.info("Generating Curation Comparison Figures in: %s", output_dir)
    log.info("-" * 75)

    # 1. Before vs After Flash Point Distribution
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df_original["flash_point"], label=f"Original Dataset (N = {len(df_original):,})",
                color="#e74c3c", linewidth=2.5, linestyle="--")
    sns.kdeplot(df_curated["flash_point"], label=f"Curated Dataset (N = {len(df_curated):,})",
                color="#27ae60", linewidth=2.5)

    plt.title("Flash Point Target Distribution: Before vs. After Outlier Curation", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Flash Point (°C)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    dist_plot_path = os.path.join(output_dir, "before_after_distribution.png")
    plt.savefig(dist_plot_path)
    plt.close()
    log.info("   Saved: %s", dist_plot_path)

    # 2. Exclusions by Stage
    if not df_all_dropped.empty and "stage" in df_all_dropped.columns:
        plt.figure(figsize=(8, 5))
        stage_counts = df_all_dropped["stage"].value_counts()
        sns.barplot(x=stage_counts.index, y=stage_counts.values, palette="mako")
        plt.title("Number of Compounds Filtered per Curation Stage", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Curation Stage", fontsize=11)
        plt.ylabel("Filtered Compounds Count", fontsize=11)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()

        stage_plot_path = os.path.join(output_dir, "filtered_counts_by_stage.png")
        plt.savefig(stage_plot_path)
        plt.close()
        log.info("   Saved: %s", stage_plot_path)

    # 3. Retention Rate by Functional Group
    if "primary_functional_group" in df_original.columns and "primary_functional_group" in df_curated.columns:
        orig_fg = df_original["primary_functional_group"].value_counts()
        cur_fg  = df_curated["primary_functional_group"].value_counts()

        top_fgs = orig_fg.head(10).index
        comp_df = pd.DataFrame({
            "Original": orig_fg[top_fgs],
            "Curated": cur_fg[top_fgs]
        }).fillna(0)
        comp_df["Retention_Rate (%)"] = (comp_df["Curated"] / comp_df["Original"]) * 100

        plt.figure(figsize=(11, 5))
        ax = sns.barplot(x=comp_df.index, y=comp_df["Retention_Rate (%)"], palette="crest")
        plt.axhline(100, color="grey", linestyle=":", alpha=0.7)
        plt.ylim(85, 102)
        plt.title("Sample Retention Rate (%) Across Top Functional Groups", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Primary Functional Group", fontsize=11)
        plt.ylabel("Retention Rate (%)", fontsize=11)
        plt.xticks(rotation=30, ha="right")

        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f"{height:.1f}%",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 2),
                        textcoords='offset points')

        plt.tight_layout()
        fg_plot_path = os.path.join(output_dir, "retention_by_functional_group.png")
        plt.savefig(fg_plot_path)
        plt.close()
        log.info("   Saved: %s", fg_plot_path)


# ======================================================================
# Section 6 -- Main Pipeline Orchestrator
# ======================================================================

def run_outlier_curation_pipeline(
    enable_radicals: bool = True,
    enable_physchem: bool = True,
    enable_statistical: bool = True,
    z_threshold: float = 3.0,
    input_featurized_csv: str = INPUT_FEATURIZED_CSV,
    input_fg_csv: str = INPUT_FG_CSV,
    output_featurized_csv: str = OUTPUT_FEATURIZED_CSV,
    output_fg_csv: str = OUTPUT_FG_CSV,
    output_dir: str = OUTPUT_DIR
) -> tuple:
    """
    Main modular execution pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_featurized_csv), exist_ok=True)

    log.info("=" * 75)
    log.info(" ChemPyQSPR -- Modular Outlier Curation & Dataset Cleaning")
    log.info(" Timestamp : %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info(" Active Curation Stages:")
    log.info("  * Stage 1 (Radical / Fragment Filter) : %s", enable_radicals)
    log.info("  * Stage 2 (Physicochemical Filter)    : %s", enable_physchem)
    log.info("  * Stage 3 (Statistical Residual >%.1f sigma): %s", z_threshold, enable_statistical)
    log.info("=" * 75)

    # Load initial dataset
    df_current, df_fg = load_datasets(input_featurized_csv, input_fg_csv)
    initial_count = len(df_current)

    all_dropped_dfs = []
    stage_summaries = {}

    # Stage 1: Radicals & Fragments
    if enable_radicals:
        df_current, dropped_s1 = filter_radical_and_fragment_species(df_current)
        if not dropped_s1.empty:
            all_dropped_dfs.append(dropped_s1)
        stage_summaries["Stage_1_Radicals_Fragments"] = {
            "enabled": True,
            "excluded_count": len(dropped_s1),
            "remaining_count": len(df_current)
        }
    else:
        stage_summaries["Stage_1_Radicals_Fragments"] = {"enabled": False, "excluded_count": 0}

    # Stage 2: Physicochemical & Unit Inconsistencies
    if enable_physchem:
        df_current, dropped_s2 = filter_physicochemical_inconsistencies(df_current)
        if not dropped_s2.empty:
            all_dropped_dfs.append(dropped_s2)
        stage_summaries["Stage_2_Physicochemical"] = {
            "enabled": True,
            "excluded_count": len(dropped_s2),
            "remaining_count": len(df_current)
        }
    else:
        stage_summaries["Stage_2_Physicochemical"] = {"enabled": False, "excluded_count": 0}

    # Stage 3: Statistical Residual Outliers (|z| > 3.0σ)
    if enable_statistical:
        df_current, dropped_s3 = filter_statistical_residual_outliers(df_current, z_threshold=z_threshold)
        if not dropped_s3.empty:
            all_dropped_dfs.append(dropped_s3)
        stage_summaries["Stage_3_Statistical_Residuals"] = {
            "enabled": True,
            "excluded_count": len(dropped_s3),
            "remaining_count": len(df_current)
        }
    else:
        stage_summaries["Stage_3_Statistical_Residuals"] = {"enabled": False, "excluded_count": 0}

    # Consolidate dropped records
    if all_dropped_dfs:
        df_all_dropped = pd.concat(all_dropped_dfs, ignore_index=True)
        # Drop duplicates by SMILES if flagged in multiple stages
        df_all_dropped = df_all_dropped.drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    else:
        df_all_dropped = pd.DataFrame()

    total_excluded = len(df_all_dropped)
    final_count = len(df_current)
    retention_pct = (final_count / initial_count) * 100

    # Align clean functional group dataset
    if df_fg is not None:
        clean_smiles_set = set(df_current["smiles"].dropna().unique())
        df_fg_clean = df_fg[df_fg["smiles"].isin(clean_smiles_set)].copy()
        df_fg_clean.to_csv(output_fg_csv, index=False)
        log.info("Saved curated functional groups CSV: %s (%d records)", output_fg_csv, len(df_fg_clean))

    # Save curated featurized dataset
    df_current.to_csv(output_featurized_csv, index=False)
    log.info("Saved curated featurized CSV       : %s (%d records x %d cols)",
             output_featurized_csv, len(df_current), df_current.shape[1])

    # Save audit log
    if not df_all_dropped.empty:
        df_all_dropped.to_csv(AUDIT_LOG_CSV, index=False)
        log.info("Saved full audit log of exclusions : %s (%d records)", AUDIT_LOG_CSV, len(df_all_dropped))

    # Save summary JSON
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "initial_compounds_count": initial_count,
        "final_curated_compounds_count": final_count,
        "total_excluded_count": total_excluded,
        "overall_retention_rate_pct": round(retention_pct, 2),
        "stages": stage_summaries,
        "output_files": {
            "curated_featurized_csv": output_featurized_csv,
            "curated_fg_csv": output_fg_csv,
            "audit_log_csv": AUDIT_LOG_CSV
        }
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    log.info("Saved curation summary JSON        : %s", SUMMARY_JSON)

    # Generate comparison plots
    df_orig_all, _ = load_datasets(input_featurized_csv, input_fg_csv)
    generate_curation_visualizations(df_orig_all, df_current, df_all_dropped, output_dir=output_dir)

    # Print Summary Report
    log.info("=" * 75)
    log.info("CURATION & OUTLIER REMOVAL COMPLETED SUCCESSFULLY!")
    log.info("=" * 75)
    log.info("Initial Valid Compounds : %d", initial_count)
    log.info("Total Outliers Excluded : %d (%.2f%%)", total_excluded, (total_excluded / initial_count) * 100)
    log.info("Final Curated Dataset   : %d (%.2f%% retention)", final_count, retention_pct)
    log.info("Target FP Range (Clean) : Min=%.2f deg C | Mean=%.2f deg C | Max=%.2f deg C | Std=%.2f deg C",
             df_current["flash_point"].min(), df_current["flash_point"].mean(),
             df_current["flash_point"].max(), df_current["flash_point"].std())
    log.info("=" * 75)

    return df_current, df_all_dropped, summary_data


# ======================================================================
# Section 7 -- CLI Argument Parser
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="ChemPyQSPR -- Modular Outlier Removal & Dataset Curation"
    )
    parser.add_argument(
        "--skip-radicals",
        action="store_true",
        default=False,
        help="Skip Stage 1: Radical & unclosed-valence fragment filtering."
    )
    parser.add_argument(
        "--skip-physchem",
        action="store_true",
        default=False,
        help="Skip Stage 2: Physicochemical consistency & Kelvin unit artifact filtering."
    )
    parser.add_argument(
        "--skip-statistical",
        action="store_true",
        default=False,
        help="Skip Stage 3: Statistical residual outlier (|z| > 3.0σ) filtering."
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=3.0,
        help="Threshold in standard deviations (|z|) for statistical outlier removal. Default: 3.0"
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=INPUT_FEATURIZED_CSV,
        help=f"Path to input featurized dataset. Default: {INPUT_FEATURIZED_CSV}"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=OUTPUT_FEATURIZED_CSV,
        help=f"Path to save curated dataset. Default: {OUTPUT_FEATURIZED_CSV}"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_outlier_curation_pipeline(
        enable_radicals=not args.skip_radicals,
        enable_physchem=not args.skip_physchem,
        enable_statistical=not args.skip_statistical,
        z_threshold=args.z_threshold,
        input_featurized_csv=args.input_csv,
        output_featurized_csv=args.output_csv
    )


if __name__ == "__main__":
    main()
