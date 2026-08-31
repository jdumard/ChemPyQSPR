"""
inference.py
============
Real-Time Flash Point Inference Module for ChemPyQSPR.

Loads the serialized champion model (best_model.pkl), feature column list
(feature_cols.pkl), and optional StandardScaler (scaler.pkl) from the
models/ directory, then accepts chemical inputs in one of two forms:
  1. SMILES string         -- directly featurized via RDKit
  2. Compound name string  -- resolved to SMILES via PubChem PUG REST API

Each input is converted to the same feature vector used during training
(RDKit physicochemical descriptors + 1024-bit Morgan Fingerprints), aligned
to the training column layout, and passed through the pipeline to obtain a
numerical Flash Point prediction in degrees C.

Modes:
  * CLI (default): pass one or more compound names / SMILES as positional args.
  * Batch CSV    : use --input-csv to process an entire file at once.
  * Interactive  : run with no arguments to enter an interactive prompt.

Usage examples:
  python src/inference.py "ethanol"
  python src/inference.py "CCO" "CC(C)O" --smiles
  python src/inference.py --input-csv data/new_compounds.csv --output-csv data/predictions.csv
  python src/inference.py   (launches interactive mode)
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
import requests
import joblib

from sklearn.pipeline import Pipeline
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

warnings.filterwarnings("ignore")

# Configure Terminal Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Paths & Constants
# ----------------------------------------------------------------------
MODEL_PATH        = "models/best_model.pkl"
FEATURE_COLS_PATH = "models/feature_cols.pkl"
SCALER_PATH       = "models/scaler.pkl"        # loaded only if present

# PubChem PUG REST endpoint — requests both IsomericSMILES and CanonicalSMILES
# to handle API version differences (field name varies by PubChem version)
PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}"
    "/property/IsomericSMILES,CanonicalSMILES/JSON"
)
REQUEST_TIMEOUT   = 10                          # seconds for PubChem API calls

FP_RADIUS         = 2                           # Morgan fingerprint radius (ECFP4)
FP_N_BITS         = 1024                        # Morgan fingerprint length

# ----------------------------------------------------------------------
# Applicability Domain (AD) Thresholds (from curated training domain)
# ----------------------------------------------------------------------
AD_MW_MAX              = 450.0                  # Max molecular weight (g/mol)
AD_MW_MIN              = 30.0                   # Min molecular weight (g/mol)
AD_HEAVY_ATOMS_MAX     = 35                     # Max heavy atom count
AD_ROTATABLE_BONDS_MAX = 15                     # Max rotatable bonds count
AD_RARE_ELEMENTS       = {                      # Heteroatoms with sparse/zero training coverage
    "Si", "P", "B", "Sn", "As", "Se", "Ge", "Sb", "Te", "Bi", "Hg", "Pb", "Ti", "Al"
}

# Physicochemical descriptors computed per molecule (must match feature_engineering.py)
RDKIT_DESCRIPTOR_NAMES = [
    "MW", "LogP", "TPSA", "HBD", "HBA",
    "RotatableBonds", "AromaticRings", "HeavyAtoms",
    "MolarRefractivity", "FractionCSP3"
]


# ======================================================================
# Section 1 -- Model Loading
# ======================================================================

def load_artifacts(
    model_path: str = MODEL_PATH,
    feature_cols_path: str = FEATURE_COLS_PATH,
    scaler_path: str = SCALER_PATH
) -> tuple:
    """
    Deserializes and returns the trained estimator, feature column list,
    and (optionally) the external StandardScaler from disk.

    Automatically detects whether the model is a bare sklearn estimator
    (e.g., RandomForest, XGBoost) or a full sklearn Pipeline (e.g., SVR,
    Ridge) that already embeds its own preprocessing steps. This distinction
    controls whether the external scaler is applied during inference.

    Parameters
    ----------
    model_path        : str -- Path to the serialized estimator or Pipeline (.pkl).
    feature_cols_path : str -- Path to the list of training feature column names (.pkl).
    scaler_path       : str -- Path to the fitted StandardScaler (.pkl).
                              Ignored automatically if the model is already a Pipeline.

    Returns
    -------
    model        : sklearn estimator or Pipeline -- The trained predictive object.
    feature_cols : list[str]                     -- Ordered list of feature column names.
    scaler       : StandardScaler | None         -- External scaler, or None if the model
                                                   is a Pipeline (scaling already embedded).
    """
    # Validate required file paths
    for path in (model_path, feature_cols_path):
        if not os.path.exists(path):
            log.error("Required artifact not found: '%s'", path)
            log.error("Ensure that train_model.py has been executed and artifacts saved to models/")
            sys.exit(1)

    log.info("Loading trained model from       : %s", model_path)
    # Try joblib first (used by sklearn's dump); fall back to pickle
    try:
        model = joblib.load(model_path)
    except Exception:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    # Introspect the model type to determine the correct inference strategy
    model_type   = type(model).__name__
    is_pipeline  = isinstance(model, Pipeline)
    log.info("Model type detected             : %s", model_type)
    log.info("Is sklearn Pipeline             : %s", is_pipeline)

    if is_pipeline:
        step_names = [s[0] for s in model.steps]
        log.info("Pipeline steps                  : %s", step_names)

    log.info("Loading feature column list from : %s", feature_cols_path)
    with open(feature_cols_path, "rb") as f:
        feature_cols = pickle.load(f)

    # Load the external scaler only for bare estimators.
    # If the model is a Pipeline, preprocessing (scaling) is already embedded
    # inside its steps (e.g., ColumnTransformer -> StandardScaler). Applying
    # an external scaler on top would double-scale continuous features.
    scaler = None
    if is_pipeline:
        log.info("Pipeline detected -- external scaler will be skipped (scaling is embedded).")
    elif os.path.exists(scaler_path):
        log.info("Bare estimator detected -- loading external scaler from: %s", scaler_path)
        try:
            scaler = joblib.load(scaler_path)
        except Exception:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        log.info("Scaler expects %d feature(s).", getattr(scaler, 'n_features_in_', '?'))
    else:
        log.info("No external scaler found at '%s'. Proceeding unscaled.", scaler_path)

    log.info("Artifacts loaded -- %d feature columns expected by the model.", len(feature_cols))
    return model, feature_cols, scaler


# ======================================================================
# Section 2 -- Chemical Featurization
# ======================================================================

def compute_rdkit_descriptors(mol: Chem.Mol) -> dict:
    """
    Computes the same 10 physicochemical descriptors used during training.
    Returns an empty dict if the Mol object is None (parsing failure).

    Parameters
    ----------
    mol : rdkit.Chem.Mol -- A valid RDKit molecule object.

    Returns
    -------
    dict mapping descriptor name to float value.
    """
    if mol is None:
        return {}

    return {
        "MW"               : Descriptors.MolWt(mol),            # Molecular weight (g/mol)
        "LogP"             : Descriptors.MolLogP(mol),          # Octanol-water partition coefficient
        "TPSA"             : Descriptors.TPSA(mol),             # Topological polar surface area (A2)
        "HBD"              : Descriptors.NumHDonors(mol),       # Number of H-bond donors
        "HBA"              : Descriptors.NumHAcceptors(mol),    # Number of H-bond acceptors
        "RotatableBonds"   : Descriptors.NumRotatableBonds(mol),# Rotatable bond count
        "AromaticRings"    : Descriptors.NumAromaticRings(mol), # Aromatic ring count
        "HeavyAtoms"       : Descriptors.HeavyAtomCount(mol),   # Heavy atom count
        "MolarRefractivity": Descriptors.MolMR(mol),            # Molar refractivity
        "FractionCSP3"     : Descriptors.FractionCSP3(mol)      # Fraction of sp3 carbons
    }


def compute_morgan_fingerprint(mol: Chem.Mol, radius: int = FP_RADIUS, n_bits: int = FP_N_BITS) -> np.ndarray:
    """
    Generates a bit-vector Morgan Fingerprint (ECFP4 when radius=2) as a
    1D NumPy integer array of length n_bits.

    Parameters
    ----------
    mol    : rdkit.Chem.Mol -- A valid RDKit molecule object.
    radius : int            -- Morgan radius (default 2 = ECFP4).
    n_bits : int            -- Fingerprint length in bits (default 1024).

    Returns
    -------
    np.ndarray of shape (n_bits,) with dtype int.
    """
    if mol is None:
        return np.zeros(n_bits, dtype=int)

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp  = gen.GetFingerprint(mol)
    return np.array(fp, dtype=int)


def smiles_to_feature_row(smiles: str) -> tuple:
    """
    Converts a SMILES string to a flat feature dictionary compatible with
    the training feature matrix layout, returning both the feature dictionary
    and the parsed RDKit Mol object.

    Parameters
    ----------
    smiles : str -- A valid SMILES string.

    Returns
    -------
    tuple of (dict mapping column name -> value, Chem.Mol object), or (None, None) if parsing fails.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        log.warning("RDKit could not parse SMILES: '%s'", smiles)
        return None, None

    record = {}

    # 1. RDKit physicochemical descriptors
    desc = compute_rdkit_descriptors(mol)
    record.update(desc)

    # 2. Morgan Fingerprint bits (prefixed 'fp_bit_N' to match training schema)
    fp_array = compute_morgan_fingerprint(mol)
    for bit_idx, bit_val in enumerate(fp_array):
        record[f"fp_bit_{bit_idx}"] = int(bit_val)

    return record, mol


# ======================================================================
# Section 3 -- Applicability Domain (AD) Engine
# ======================================================================

def evaluate_applicability_domain(mol: Chem.Mol, descriptors: dict) -> tuple:
    """
    Evaluates whether a molecule resides inside the curated training domain
    of ChemPyQSPR. Assesses:
      0. Organic compound scope (carbon must be present -- hard gate)
      1. Molecular weight limits
      2. Presence of rare/underrepresented heteroatoms (e.g. Si, P, B)
      3. Heavy atom count
      4. Unstable free radical states

    Parameters
    ----------
    mol         : rdkit.Chem.Mol -- Parsed RDKit molecule object.
    descriptors : dict           -- Physicochemical descriptor mapping.

    Returns
    -------
    in_domain    : bool      -- True if inside domain; False if out of domain.
    warnings     : list[str] -- Domain violation messages (empty when in_domain is True).
    is_inorganic : bool      -- True when the compound contains no carbon atoms
                               (hard out-of-scope, prediction is blocked entirely).
    """
    if mol is None:
        return False, ["Invalid molecular structure (cannot evaluate applicability domain)"], False

    domain_warnings = []

    # 0. Organic Compound Scope Guard (HARD GATE)
    # ChemPyQSPR was trained exclusively on organic compounds.  Any molecule
    # that contains no carbon atom is outside the model's scope and CANNOT be
    # predicted.  This check is evaluated first and returns immediately.
    present_elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
    if "C" not in present_elements:
        elem_list = ", ".join(sorted(present_elements)) if present_elements else "(none)"
        scope_msg = (
            f"INORGANIC COMPOUND -- out of model scope. "
            f"ChemPyQSPR was trained exclusively on organic compounds (carbon-containing). "
            f"This molecule contains no carbon atom (elements found: {elem_list}). "
            f"Flash point prediction is not available for inorganic species."
        )
        return False, [scope_msg], True

    # 1. Molecular Weight Threshold Check
    mw = descriptors.get("MW", Descriptors.MolWt(mol))
    if mw > AD_MW_MAX:
        domain_warnings.append(
            f"High Molecular Weight ({mw:.1f} g/mol > {AD_MW_MAX:.0f} g/mol threshold) - model is extrapolating"
        )
    elif mw < AD_MW_MIN:
        domain_warnings.append(
            f"Low Molecular Weight ({mw:.1f} g/mol < {AD_MW_MIN:.0f} g/mol threshold) - ultra-light volatile risk"
        )

    # 2. Rare / Underrepresented Heteroatoms Check (e.g. Si, P, B, metals)
    rare_found = present_elements.intersection(AD_RARE_ELEMENTS)
    if rare_found:
        elem_list = ", ".join(sorted(rare_found))
        domain_warnings.append(
            f"Contains rare/underrepresented heteroatom(s): [{elem_list}] - lacking sufficient training support"
        )

    # 3. Heavy Atom Count Check
    heavy_atoms = descriptors.get("HeavyAtoms", Descriptors.HeavyAtomCount(mol))
    if heavy_atoms > AD_HEAVY_ATOMS_MAX:
        domain_warnings.append(
            f"High Heavy Atom Count ({heavy_atoms} > {AD_HEAVY_ATOMS_MAX}) - structurally outside training space"
        )

    # 4. Free Radical Check
    num_radicals = sum([atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()])
    if num_radicals > 0:
        domain_warnings.append(
            f"Unstable free radical species ({num_radicals} unpaired electron(s))"
        )

    in_domain = len(domain_warnings) == 0
    return in_domain, domain_warnings, False


# ======================================================================
# Section 3 -- Compound Name Resolution (PubChem)
# ======================================================================

def resolve_name_to_smiles(compound_name: str) -> str:
    """
    Queries the PubChem PUG REST API to resolve a compound name (IUPAC,
    trade name, synonym, CAS, etc.) into a canonical SMILES string.

    Parameters
    ----------
    compound_name : str -- Human-readable compound name or identifier.

    Returns
    -------
    str (SMILES) if the name was resolved successfully, or None on failure.
    """
    url = PUBCHEM_URL.format(name=requests.utils.quote(compound_name))
    log.info("   >> Querying PubChem for '%s'...", compound_name)

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data  = response.json()
        props = data["PropertyTable"]["Properties"][0]

        # PubChem may return the key as "IsomericSMILES", "CanonicalSMILES",
        # or simply "SMILES" depending on the API version and compound type.
        # Try all known key names in order of preference.
        smiles = (
            props.get("IsomericSMILES")
            or props.get("CanonicalSMILES")
            or props.get("SMILES")
        )

        if not smiles:
            log.warning("   PubChem response has no SMILES field for '%s'. Keys: %s",
                        compound_name, list(props.keys()))
            return None

        log.info("   Resolved SMILES : %s", smiles)
        return smiles

    except requests.exceptions.Timeout:
        log.warning("   PubChem request timed out for '%s'", compound_name)
    except requests.exceptions.HTTPError as e:
        log.warning("   PubChem HTTP error for '%s': %s", compound_name, e)
    except (KeyError, IndexError) as e:
        log.warning("   PubChem parse error for '%s': %s", compound_name, e)
    except ValueError as e:
        log.warning("   PubChem JSON decode error for '%s': %s", compound_name, e)
    except requests.exceptions.RequestException as e:
        log.warning("   Network error while querying PubChem: %s", e)

    return None


# ======================================================================
# Section 4 -- Feature Alignment
# ======================================================================

def align_features(raw_features: dict, feature_cols: list) -> pd.DataFrame:
    """
    Converts a raw feature dictionary into a single-row DataFrame whose
    columns exactly match the training feature layout. Missing columns are
    filled with 0 (safe default for binary fingerprint bits); extra columns
    from the raw dict are silently dropped.

    Parameters
    ----------
    raw_features : dict       -- Feature name to value mapping from featurization.
    feature_cols : list[str]  -- Ordered column list from training (feature_cols.pkl).

    Returns
    -------
    pd.DataFrame of shape (1, len(feature_cols)).
    """
    row_df = pd.DataFrame([raw_features])

    # Add any missing columns with a safe default of 0
    missing_cols = [c for c in feature_cols if c not in row_df.columns]
    if missing_cols:
        log.debug("   Padding %d missing feature columns with 0.", len(missing_cols))
        for col in missing_cols:
            row_df[col] = 0

    # Reorder columns to exactly match the training schema
    return row_df[feature_cols]


# ======================================================================
# Section 5 -- Core Inference Logic
# ======================================================================

def predict_flash_point(
    input_value: str,
    model,
    feature_cols: list,
    scaler=None,
    is_smiles: bool = False
) -> dict:
    """
    End-to-end inference for a single compound: resolves the input, featurizes
    it, aligns features, and returns a prediction dictionary.

    Parameters
    ----------
    input_value  : str            -- Compound name or SMILES string.
    model        : sklearn estimator -- Loaded predictive pipeline.
    feature_cols : list[str]      -- Training feature column names.
    scaler       : StandardScaler | None -- External scaler (or None if embedded).
    is_smiles    : bool           -- If True, treat input_value directly as SMILES.

    Returns
    -------
    dict with keys:
        input        -- Original user input.
        smiles       -- Resolved or provided SMILES string (or None).
        predicted_fp -- Predicted flash point in degrees C (float, or None on failure).
        status       -- "success", "smiles_error", or "resolution_error".
        timestamp    -- ISO 8601 inference timestamp.
    """
    result = {
        "input"                  : input_value,
        "smiles"                 : None,
        "predicted_fp"           : None,
        "in_applicability_domain": None,
        "domain_warnings"        : [],
        "status"                 : None,
        "timestamp"              : datetime.now().isoformat()
    }

    # Step 1: Obtain SMILES
    if is_smiles:
        smiles = input_value
        result["smiles"] = smiles
    else:
        smiles = resolve_name_to_smiles(input_value)
        result["smiles"] = smiles

    if smiles is None:
        result["status"] = "resolution_error"
        return result

    # Step 2: Featurize SMILES into descriptor + fingerprint vector
    raw_features, mol_obj = smiles_to_feature_row(smiles)
    if raw_features is None:
        result["status"] = "smiles_error"
        return result

    # Step 3: Applicability Domain Evaluation
    # evaluate_applicability_domain now returns a third flag (is_inorganic) that
    # acts as a hard gate: inorganic compounds are out of scope and cannot be
    # predicted regardless of any other domain criteria.
    in_domain, ad_warnings, is_inorganic = evaluate_applicability_domain(mol_obj, raw_features)
    result["in_applicability_domain"] = in_domain
    result["domain_warnings"]         = ad_warnings

    if is_inorganic:
        # Hard gate: block prediction. The caller is responsible for
        # displaying the user-facing message (interactive, CLI, or batch).
        result["status"] = "out_of_scope"
        return result

    # Step 4: Align feature vector to training schema
    X_input = align_features(raw_features, feature_cols)

    # Step 5: Apply external scaler (bare estimators only).
    # When scaler is None it means either:
    #   (a) the model is a Pipeline with preprocessing embedded, or
    #   (b) no scaler artifact was found on disk.
    # In both cases X_input is passed directly to the model.
    # When scaler is provided, validate that its expected feature count matches
    # the input shape before transforming, to catch any schema mismatch.
    if scaler is not None:
        expected_n = getattr(scaler, 'n_features_in_', None)
        actual_n   = X_input.shape[1]
        if expected_n is not None and expected_n != actual_n:
            log.warning(
                "   Scaler expects %d features but input has %d. "
                "Skipping external scaling to avoid corrupting the feature vector.",
                expected_n, actual_n
            )
        else:
            try:
                X_input_scaled = scaler.transform(X_input)
                X_input = pd.DataFrame(X_input_scaled, columns=feature_cols)
            except Exception as e:
                log.warning("   External scaler transform failed: %s. Proceeding unscaled.", e)

    # Step 6: Predict -- model.predict() works for any sklearn-compatible estimator
    # or Pipeline (Ridge, SVR, RF, XGBoost, etc.) without modification.
    predicted_value = float(model.predict(X_input)[0])

    result["predicted_fp"] = round(predicted_value, 2)
    result["status"]        = "success"

    return result


# ======================================================================
# Section 6 -- Batch Inference from CSV
# ======================================================================

def run_batch_inference(
    input_csv: str,
    output_csv: str,
    model,
    feature_cols: list,
    scaler=None,
    smiles_col: str = "smiles",
    name_col: str = "name"
) -> pd.DataFrame:
    """
    Processes a CSV file row-by-row and appends predicted flash point values.
    The function tries to use the 'smiles' column first; if absent, it falls
    back to resolving compound names via PubChem.

    Parameters
    ----------
    input_csv    : str        -- Path to the input CSV file.
    output_csv   : str        -- Path where the enriched CSV will be saved.
    model        : estimator  -- Loaded model pipeline.
    feature_cols : list[str]  -- Training feature columns.
    scaler       : scaler | None -- External scaler (or None).
    smiles_col   : str        -- Column name for SMILES in the input CSV.
    name_col     : str        -- Column name for compound names in the input CSV.

    Returns
    -------
    pd.DataFrame -- Input data enriched with predicted flash point column.
    """
    if not os.path.exists(input_csv):
        log.error("Batch input file not found: '%s'", input_csv)
        sys.exit(1)

    df_input = pd.read_csv(input_csv)
    log.info("Batch mode -- Loaded %d compounds from: %s", len(df_input), input_csv)

    predictions = []
    domain_flags = []
    domain_warnings_col = []

    for idx, row in df_input.iterrows():
        log.info("-" * 60)
        log.info("Compound %d / %d", idx + 1, len(df_input))

        # Determine input type (SMILES vs name)
        if smiles_col in df_input.columns and pd.notna(row.get(smiles_col)):
            compound_input = str(row[smiles_col])
            use_smiles     = True
            display_label  = f"SMILES: {compound_input}"
        elif name_col in df_input.columns and pd.notna(row.get(name_col)):
            compound_input = str(row[name_col])
            use_smiles     = False
            display_label  = f"Name  : {compound_input}"
        else:
            log.warning("   Row %d has neither a valid SMILES nor a compound name. Skipping.", idx)
            predictions.append(None)
            domain_flags.append(None)
            domain_warnings_col.append("Skipped: missing SMILES and name")
            continue

        log.info("   Input : %s", display_label)
        pred_result = predict_flash_point(
            compound_input, model, feature_cols, scaler, is_smiles=use_smiles
        )

        fp_val = pred_result["predicted_fp"]
        in_dom = pred_result["in_applicability_domain"]
        w_text = "; ".join(pred_result["domain_warnings"]) if pred_result["domain_warnings"] else "None"

        predictions.append(fp_val)
        domain_flags.append(in_dom)
        domain_warnings_col.append(w_text)

        if fp_val is not None:
            dom_str = "[In Domain]" if in_dom else "[OUT OF DOMAIN WARNING]"
            log.info("   Predicted Flash Point : %.2f degrees C %s", fp_val, dom_str)
        else:
            log.warning("   Prediction failed with status: %s", pred_result["status"])

    # Attach prediction and applicability domain columns
    df_input["predicted_flash_point_C"] = predictions
    df_input["in_applicability_domain"] = domain_flags
    df_input["domain_warnings"]         = domain_warnings_col

    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    df_input.to_csv(output_csv, index=False)
    log.info("=" * 60)
    log.info("Batch inference complete. Results saved to: %s", output_csv)

    return df_input


# ======================================================================
# Section 7 -- Interactive Mode
# ======================================================================

def run_interactive_mode(model, feature_cols: list, scaler=None) -> None:
    """
    Launches an interactive REPL loop where the user can type compound names
    or SMILES strings one at a time and receive flash point predictions with
    applicability domain verification.
    Type 'exit' or 'quit' to terminate the session.

    Parameters
    ----------
    model        : estimator -- Loaded model pipeline.
    feature_cols : list[str] -- Training feature columns.
    scaler       : scaler | None -- External scaler (or None).
    """
    # Silence INFO/WARNING logs during the interactive session so that only
    # user-facing print() messages appear in the terminal. Unexpected technical
    # errors (e.g. scaler mismatch) are surfaced via log.warning.
    logging.getLogger().setLevel(logging.ERROR)

    print("\n" + "=" * 68)
    print("  ChemPyQSPR -- Interactive Flash Point Prediction & Domain Check")
    print("=" * 68)
    print("  Enter a compound name (e.g., 'acetone') or SMILES string.")
    print("  Prefix SMILES with 's:' (e.g., 's:CCO') to skip PubChem.")
    print("  Type 'exit' or 'quit' to end the session.")
    print("=" * 68 + "\n")

    session_results = []

    while True:
        try:
            user_input = input("Input compound >> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession terminated.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Exiting interactive session.")
            break

        # Detect explicit SMILES prefix 's:'
        if user_input.lower().startswith("s:"):
            compound  = user_input[2:].strip()
            is_smiles = True
        else:
            compound  = user_input
            is_smiles = False

        result = predict_flash_point(compound, model, feature_cols, scaler, is_smiles=is_smiles)
        session_results.append(result)

        print()
        if result["status"] == "out_of_scope":
            print(f"  ! Compound     : {result['input']}")
            print(f"  ! SMILES       : {result['smiles']}")
            print("  ! [OUT OF SCOPE] This compound is INORGANIC (no carbon atoms detected).")
            print("  ! ChemPyQSPR was trained exclusively on organic compounds.")
            print("  ! Flash point prediction is NOT available for inorganic species.")
            for w in result["domain_warnings"]:
                print(f"    * {w}")
        elif result["status"] == "success":
            print(f"  + Compound     : {result['input']}")
            print(f"  + SMILES       : {result['smiles']}")
            print(f"  + Flash Point  : {result['predicted_fp']:.2f} degrees C")
            if result["in_applicability_domain"]:
                print("  + Domain Status: [IN DOMAIN] High Confidence")
            else:
                print("  ! Domain Status: [OUT OF DOMAIN WARNING] Extrapolation Risk")
                for w in result["domain_warnings"]:
                    print(f"    * Warning: {w}")
        else:
            print(f"  ! Prediction failed for '{result['input']}' -- Status: {result['status']}")
        print()

    # Summary table at end of session
    if session_results:
        print("\n" + "=" * 68)
        print("  Session Summary")
        print("=" * 68)
        summary_rows = []
        for r in session_results:
            # For inorganic (out_of_scope) compounds the full warning text is
            # already shown during the per-compound output above. In the summary
            # table we only display the concise label "out of scope" to keep it
            # readable, while other statuses show their domain warnings normally.
            if r["status"] == "out_of_scope":
                warn_text = "out of scope"
            else:
                warn_text = "; ".join(r.get("domain_warnings", [])) or "None"
            summary_rows.append({
                "input": r["input"],
                "predicted_fp": r["predicted_fp"],
                "in_domain": r.get("in_applicability_domain"),
                "status": r["status"],
                "warnings": warn_text
            })
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))
        print("=" * 68 + "\n")

    # Restore log level so CLI and batch modes can log normally.
    logging.getLogger().setLevel(logging.INFO)


# ======================================================================
# Section 8 -- CLI Entry Point
# ======================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Defines and parses command-line arguments for all three execution modes:
    single/multi-compound CLI, batch CSV, and interactive.
    """
    parser = argparse.ArgumentParser(
        prog="inference.py",
        description="ChemPyQSPR -- Real-Time Flash Point Prediction Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/inference.py ethanol methanol\n"
            "  python src/inference.py CCO CC(C)O --smiles\n"
            "  python src/inference.py --input-csv data/new.csv --output-csv data/pred.csv\n"
            "  python src/inference.py   (launches interactive mode)\n"
        )
    )

    parser.add_argument(
        "compounds",
        nargs="*",
        help="One or more compound names or SMILES strings to predict."
    )
    parser.add_argument(
        "--smiles",
        action="store_true",
        default=False,
        help="Treat all positional arguments as SMILES (skip PubChem lookup)."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to a CSV file for batch prediction."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/processed/inference_predictions.csv",
        help="Destination path for the batch prediction output CSV."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help=f"Path to the serialized model (.pkl). Default: {MODEL_PATH}"
    )
    parser.add_argument(
        "--feature-cols",
        type=str,
        default=FEATURE_COLS_PATH,
        help=f"Path to feature column list (.pkl). Default: {FEATURE_COLS_PATH}"
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default=SCALER_PATH,
        help=f"Path to the external StandardScaler (.pkl). Default: {SCALER_PATH}"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="If set, save CLI prediction results as a JSON file at this path."
    )

    return parser.parse_args()


def main() -> None:
    """
    Main dispatcher -- routes execution to CLI, batch, or interactive mode
    based on provided arguments.
    """
    args = parse_arguments()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    log.info("=" * 70)
    log.info(" ChemPyQSPR -- Real-Time Flash Point Inference Module")
    log.info(" Timestamp : %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 70)

    # ------------------------------------------------------------------
    # Load Model Artifacts
    # ------------------------------------------------------------------
    model, feature_cols, scaler = load_artifacts(
        model_path=args.model,
        feature_cols_path=args.feature_cols,
        scaler_path=args.scaler
    )

    # ------------------------------------------------------------------
    # Mode A: Batch CSV Inference
    # ------------------------------------------------------------------
    if args.input_csv:
        log.info("Mode: Batch CSV")
        run_batch_inference(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            model=model,
            feature_cols=feature_cols,
            scaler=scaler
        )
        return

    # ------------------------------------------------------------------
    # Mode B: CLI -- One or More Compounds
    # ------------------------------------------------------------------
    if args.compounds:
        log.info("Mode: CLI -- %d compound(s) to process", len(args.compounds))

        cli_results = []

        for compound in args.compounds:
            log.info("-" * 60)
            log.info("Processing: %s", compound)
            result = predict_flash_point(
                compound, model, feature_cols, scaler, is_smiles=args.smiles
            )
            cli_results.append(result)

            if result["status"] == "out_of_scope":
                log.warning(
                    "   * [OUT OF SCOPE] Inorganic compound -- flash point prediction blocked. "
                    "ChemPyQSPR only supports organic (carbon-containing) compounds."
                )
            elif result["status"] == "success":
                dom_label = "[IN DOMAIN]" if result["in_applicability_domain"] else "[OUT OF DOMAIN WARNING]"
                log.info("   * Predicted Flash Point : %.2f degrees C %s", result["predicted_fp"], dom_label)
            else:
                log.warning("   * Prediction failed -- Status: %s", result["status"])

        # Print summary table
        log.info("=" * 70)
        log.info("CLI Prediction & Applicability Domain Summary:")
        summary_rows = []
        for r in cli_results:
            summary_rows.append({
                "input"       : r["input"],
                "smiles"      : r["smiles"],
                "predicted_fp": r["predicted_fp"],
                "in_domain"   : r["in_applicability_domain"],
                "status"      : r["status"],
                "warnings"    : "; ".join(r["domain_warnings"]) if r["domain_warnings"] else "None"
            })
        summary_df = pd.DataFrame(summary_rows)
        log.info("\n\n%s\n", summary_df.to_string(index=False))

        # Optionally save results to JSON
        if args.output_json:
            os.makedirs(os.path.dirname(args.output_json) if os.path.dirname(args.output_json) else ".", exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp" : datetime.now().isoformat(),
                    "mode"      : "cli",
                    "n_inputs"  : len(args.compounds),
                    "results"   : cli_results
                }, f, indent=2, default=str)
            log.info("Results saved to JSON: %s", args.output_json)

        log.info("=" * 70)
        return

    # ------------------------------------------------------------------
    # Mode C: Interactive (no arguments provided)
    # ------------------------------------------------------------------
    run_interactive_mode(model, feature_cols, scaler)


# ======================================================================
# Script Entry Point
# ======================================================================

if __name__ == "__main__":
    main()

