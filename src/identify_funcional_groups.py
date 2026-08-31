"""
identify_funcional_groups.py
============================
Structural identification of organic functional groups in chemical datasets
using RDKit SMARTS patterns based on SMILES representations.
"""

# ----------------------------------------------------------------------
# Imports & Configuration
# ----------------------------------------------------------------------
import os
import sys
import logging
import pandas as pd
from rdkit import Chem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Input and Output Paths Configuration
# ----------------------------------------------------------------------
INPUT_CSV = "data/processed/flash_point_database.csv"
OUTPUT_CSV = "data/processed/cleaned_flash_point_database_fg.csv"

# ----------------------------------------------------------------------
# SMARTS Patterns Dictionary for Organic Functional Groups
# ----------------------------------------------------------------------
# Order of priority helps in classifying polyfunctional compounds
SMARTS_PATTERNS = {
    "carboxylic_acid": Chem.MolFromSmarts("[CX3](=O)[OX2H1]"),
    "ester": Chem.MolFromSmarts("[CX3](=O)[OX2H0][#6]"),
    "aldehyde": Chem.MolFromSmarts("[CX3H1](=O)[#6,H]"),
    "ketone": Chem.MolFromSmarts("[#6][CX3](=O)[#6]"),
    "alcohol": Chem.MolFromSmarts("[#6][OX2H]"),
    "amine": Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O)]"),
    "ether": Chem.MolFromSmarts("[OD2]([#6])[#6]"),
    "organohalide": Chem.MolFromSmarts("[#6][F,Cl,Br,I]"),
    "aromatic": Chem.MolFromSmarts("a"),
    "nitrile": Chem.MolFromSmarts("[NX1]#[CX2]"),
    "nitro": Chem.MolFromSmarts("[$([NX3](=O)=O),$([NX3+](=O)[O-])]")
}


# ----------------------------------------------------------------------
# Structural Identification Functions
# ----------------------------------------------------------------------
def identify_all_functional_groups(smiles: str) -> str:
    """
    Returns a semicolon-separated string of all functional groups
    identified in the molecule (useful for polyfunctional compounds).
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return "Invalid"

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return "Invalid_SMILES"

    detected_groups = []
    for group_name, pattern in SMARTS_PATTERNS.items():
        if pattern and mol.HasSubstructMatch(pattern):
            detected_groups.append(group_name)

    # If no heteroatoms or specific functional groups are matched, classify as Hydrocarbon
    if not detected_groups:
        return "hydrocarbon"

    return "; ".join(detected_groups)


def identify_primary_functional_group(smiles: str) -> str:
    """
    Returns only the highest-priority functional group for single-class
    assignment, grouping, or stratified cross-validation.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return "Invalid"

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return "Invalid_SMILES"

    for group_name, pattern in SMARTS_PATTERNS.items():
        if pattern and mol.HasSubstructMatch(pattern):
            return group_name

    return "hydrocarbon"


# ----------------------------------------------------------------------
# Pipeline Execution and Processing
# ----------------------------------------------------------------------
def main():
    log.info("=" * 70)
    log.info("ORGANIC FUNCTIONAL GROUP IDENTIFICATION VIA RDKit (SMARTS)")
    log.info("=" * 70)

    if not os.path.exists(INPUT_CSV):
        log.error("Input file not found: %s", INPUT_CSV)
        sys.exit(1)

    log.info("Loading dataset from: %s", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)

    # Verify and map required columns
    required_columns = ['name', 'smiles', 'flash_point']
    for col in required_columns:
        if col not in df.columns:
            matching_cols = [c for c in df.columns if col in c.lower()]
            if matching_cols:
                df.rename(columns={matching_cols[0]: col}, inplace=True)

    log.info("Identifying functional groups for N = %d compounds...", len(df))

    # Apply identification functions row-wise
    df['primary_functional_group'] = df['smiles'].apply(identify_primary_functional_group)
    df['all_functional_groups'] = df['smiles'].apply(identify_all_functional_groups)

    # Save enriched dataset to CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    log.info("=" * 70)
    log.info("PROCESSING COMPLETED SUCCESSFULLY!")
    log.info("=" * 70)
    log.info("Output file saved at: %s", OUTPUT_CSV)
    log.info("Compound Distribution by Primary Functional Group:\n%s", df['primary_functional_group'].value_counts().to_string())
    log.info("Preview of Results (First 10 rows):\n%s", df[['name', 'smiles', 'flash_point', 'primary_functional_group']].head(10).to_string(index=False))
    log.info("=" * 70)


if __name__ == "__main__":
    main()