"""
feature_engineering.py
======================
Calculates physicochemical descriptors (RDKit) and Morgan Fingerprints (ECFP4)
for organic compounds based on canonical SMILES representations.
"""

# ----------------------------------------------------------------------
# Imports & Configuration
# ----------------------------------------------------------------------
import os
import sys
import logging
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Input and Output Paths Configuration
# ----------------------------------------------------------------------
INPUT_CSV = "data/processed/cleaned_flash_point_database_fg.csv"
OUTPUT_CSV = "data/processed/featurized_dataset.csv"


# ----------------------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------------------
def load_dataset(input_path: str) -> pd.DataFrame:
    """Reads the processed dataset from the ETL pipeline."""
    if not os.path.exists(input_path):
        log.error("File not found at: %s", input_path)
        sys.exit(1)

    df = pd.read_csv(input_path)
    log.info("Dataset loaded: %d records.", len(df))
    return df


# ----------------------------------------------------------------------
# Calculate RDKit Descriptors
# ----------------------------------------------------------------------
def calculate_rdkit_descriptors(mol: Chem.Mol) -> dict:
    """
    Calculates a core set of 10 continuous physicochemical descriptors
    from an RDKit Mol object.
    """
    if mol is None:
        return {}

    return {
        'MW': Descriptors.MolWt(mol),                # Molecular weight
        'LogP': Descriptors.MolLogP(mol),            # Octanol-water partition coefficient
        'TPSA': Descriptors.TPSA(mol),              # Topological polar surface area
        'HBD': Descriptors.NumHDonors(mol),          # Number of hydrogen bond donors
        'HBA': Descriptors.NumHAcceptors(mol),        # Number of hydrogen bond acceptors
        'RotatableBonds': Descriptors.NumRotatableBonds(mol), # Number of rotatable bonds
        'AromaticRings': Descriptors.NumAromaticRings(mol),   # Number of aromatic rings
        'HeavyAtoms': Descriptors.HeavyAtomCount(mol),        # Number of heavy atoms
        'MolarRefractivity': Descriptors.MolMR(mol),          # Molar refractivity
        'FractionCSP3': Descriptors.FractionCSP3(mol)         # Fraction of sp3 hybridized carbon atoms
    }


# ----------------------------------------------------------------------
# Generate Morgan Fingerprint
# ----------------------------------------------------------------------
def generate_morgan_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """
    Generates a Morgan Fingerprint (ECFP4 if radius=2) of length n_bits as a NumPy array.
    """
    if mol is None:
        return np.zeros(n_bits, dtype=int)

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = mfpgen.GetFingerprint(mol)
    return np.array(fp)


# ----------------------------------------------------------------------
# Pipeline Execution and Processing
# ----------------------------------------------------------------------
def run_feature_engineering_pipeline(
    input_file: str = INPUT_CSV,
    output_file: str = OUTPUT_CSV
):
    """
    Complete Chemical Featurization Pipeline for ChemPyQSPR.
    """
    log.info("=" * 70)
    log.info("STARTING STEP 3: FEATURE ENGINEERING (FEATURIZATION)")
    log.info("=" * 70)

    df = load_dataset(input_file)

    valid_records = []
    failed_smiles = 0

    log.info("Converting SMILES to RDKit Mol objects and extracting features...")

    for idx, row in df.iterrows():
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            failed_smiles += 1
            log.warning("Unable to parse SMILES for record %s: %s", row.get('name', idx), smiles)
            continue

        # 1. Metadata and Target Variable
        record = {
            'name': row.get('name'),
            'smiles': smiles,
            'flash_point': row.get('flash_point'),  # Target Y
            'primary_functional_group': row.get('primary_functional_group'),
            'all_functional_groups': row.get('all_functional_groups')
        }

        # 2. RDKit Descriptors
        desc_dict = calculate_rdkit_descriptors(mol)
        record.update(desc_dict)

        # 3. Morgan Fingerprints (1024 bits)
        fp_array = generate_morgan_fingerprint(mol, radius=2, n_bits=1024)
        for bit_idx, bit_val in enumerate(fp_array):
            record[f'fp_bit_{bit_idx}'] = bit_val

        valid_records.append(record)

    featured_df = pd.DataFrame(valid_records)

    # Save output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    featured_df.to_csv(output_file, index=False)

    log.info("=" * 70)
    log.info("FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
    log.info("Records processed and featurized: %d / %d", len(featured_df), len(df))
    if failed_smiles > 0:
        log.warning("Invalid SMILES ignored: %d", failed_smiles)
    log.info("Feature matrix dimensions: %s (compounds x features)", str(featured_df.shape))
    log.info("File saved at: %s", output_file)
    log.info("=" * 70)


if __name__ == "__main__":
    run_feature_engineering_pipeline()