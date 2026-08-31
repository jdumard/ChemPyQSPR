"""
predict_interactive.py
======================
Streamlined Interactive Flash Point Inference Module for ChemPyQSPR.

Loads the serialized champion model (best_model.pkl), feature column list
(feature_cols.pkl), and optional StandardScaler (scaler.pkl) from the
models/ directory, then immediately launches an interactive terminal session.

The user can type:
  - A compound name (e.g., 'acetone')     -- resolved to SMILES via PubChem
  - A SMILES string prefixed with 's:'    -- e.g., 's:CCO' (skips PubChem)
  - 'exit' or 'quit'                      -- terminates the session

Usage:
    python src/predict_interactive.py
"""

# ----------------------------------------------------------------------
# Imports & Configuration
# ----------------------------------------------------------------------
import os
import sys
import pickle
import logging
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

# PubChem PUG REST endpoint
PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}"
    "/property/IsomericSMILES,CanonicalSMILES/JSON"
)
REQUEST_TIMEOUT = 10   # seconds for PubChem API calls

FP_RADIUS = 2          # Morgan fingerprint radius (ECFP4)
FP_N_BITS = 1024       # Morgan fingerprint length

# ----------------------------------------------------------------------
# Applicability Domain (AD) Thresholds (from curated training domain)
# ----------------------------------------------------------------------
AD_MW_MAX          = 450.0                  # Max molecular weight (g/mol)
AD_MW_MIN          = 30.0                   # Min molecular weight (g/mol)
AD_HEAVY_ATOMS_MAX = 35                     # Max heavy atom count
AD_RARE_ELEMENTS   = {                      # Heteroatoms with sparse/zero training coverage
    "Si", "P", "B", "Sn", "As", "Se", "Ge", "Sb", "Te", "Bi", "Hg", "Pb", "Ti", "Al"
}


# ======================================================================
# Section 1 -- Model Loading
# ======================================================================

def load_artifacts() -> tuple:
    """
    Deserializes and returns the trained estimator, feature column list,
    and (optionally) the external StandardScaler from disk.

    Automatically detects whether the model is a bare sklearn estimator
    or a full sklearn Pipeline that already embeds its own preprocessing
    steps, controlling whether the external scaler is applied at inference.

    Returns
    -------
    model        : sklearn estimator or Pipeline -- The trained predictive object.
    feature_cols : list[str]                     -- Ordered feature column names.
    scaler       : StandardScaler | None         -- External scaler, or None.
    """
    for path in (MODEL_PATH, FEATURE_COLS_PATH):
        if not os.path.exists(path):
            log.error("Required artifact not found: '%s'", path)
            log.error("Ensure that train_model.py has been executed and artifacts saved to models/")
            sys.exit(1)

    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

    is_pipeline = isinstance(model, Pipeline)
    log.info("Model type detected : %s  |  Is Pipeline: %s", type(model).__name__, is_pipeline)

    with open(FEATURE_COLS_PATH, "rb") as f:
        feature_cols = pickle.load(f)
    log.info("Feature columns loaded: %d columns", len(feature_cols))

    # Load the external scaler only for bare estimators.
    # If the model is a Pipeline, preprocessing is already embedded inside its
    # steps. Applying an external scaler on top would double-scale features.
    scaler = None
    if not is_pipeline and os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception:
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)
        log.info("External scaler loaded from: %s", SCALER_PATH)
    elif is_pipeline:
        log.info("Pipeline detected -- external scaler will be skipped (scaling is embedded).")

    return model, feature_cols, scaler


# ======================================================================
# Section 2 -- Chemical Featurization
# ======================================================================

def _compute_descriptors(mol: Chem.Mol) -> dict:
    """
    Computes the same 10 physicochemical descriptors used during training.

    Parameters
    ----------
    mol : rdkit.Chem.Mol -- A valid RDKit molecule object.

    Returns
    -------
    dict mapping descriptor name to float value.
    """
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
        "FractionCSP3"     : Descriptors.FractionCSP3(mol),     # Fraction of sp3 carbons
    }


def _compute_fingerprint(mol: Chem.Mol) -> np.ndarray:
    """
    Generates a bit-vector Morgan Fingerprint (ECFP4) as a 1D NumPy
    integer array of length FP_N_BITS.

    Parameters
    ----------
    mol : rdkit.Chem.Mol -- A valid RDKit molecule object.

    Returns
    -------
    np.ndarray of shape (FP_N_BITS,) with dtype int.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_N_BITS)
    return np.array(gen.GetFingerprint(mol), dtype=int)


def smiles_to_features(smiles: str) -> tuple:
    """
    Converts a SMILES string to a flat feature dictionary compatible with
    the training feature matrix layout, returning both the feature dict
    and the parsed RDKit Mol object.

    Parameters
    ----------
    smiles : str -- A valid SMILES string.

    Returns
    -------
    tuple of (dict, Chem.Mol), or (None, None) if parsing fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        log.warning("RDKit could not parse SMILES: '%s'", smiles)
        return None, None

    record = _compute_descriptors(mol)
    for i, bit in enumerate(_compute_fingerprint(mol)):
        record[f"fp_bit_{i}"] = int(bit)

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
    # ChemPyQSPR was trained exclusively on organic compounds. Any molecule
    # that contains no carbon atom is outside the model's scope and CANNOT
    # be predicted. This check is evaluated first and returns immediately.
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
        domain_warnings.append(
            f"Contains rare/underrepresented heteroatom(s): [{', '.join(sorted(rare_found))}] "
            f"- lacking sufficient training support"
        )

    # 3. Heavy Atom Count Check
    heavy_atoms = descriptors.get("HeavyAtoms", Descriptors.HeavyAtomCount(mol))
    if heavy_atoms > AD_HEAVY_ATOMS_MAX:
        domain_warnings.append(
            f"High Heavy Atom Count ({heavy_atoms} > {AD_HEAVY_ATOMS_MAX}) - structurally outside training space"
        )

    # 4. Free Radical Check
    num_radicals = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
    if num_radicals > 0:
        domain_warnings.append(
            f"Unstable free radical species ({num_radicals} unpaired electron(s))"
        )

    in_domain = len(domain_warnings) == 0
    return in_domain, domain_warnings, False


# ======================================================================
# Section 4 -- Compound Name Resolution (PubChem)
# ======================================================================

def resolve_name_to_smiles(compound_name: str):
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
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        props = resp.json()["PropertyTable"]["Properties"][0]
        smiles = (
            props.get("IsomericSMILES")
            or props.get("CanonicalSMILES")
            or props.get("SMILES")
        )
        if smiles:
            log.info("   Resolved SMILES : %s", smiles)
        else:
            log.warning("   PubChem response has no SMILES field for '%s'.", compound_name)
        return smiles
    except requests.exceptions.Timeout:
        log.warning("   PubChem request timed out for '%s'.", compound_name)
    except requests.exceptions.HTTPError as e:
        log.warning("   PubChem HTTP error for '%s': %s", compound_name, e)
    except (KeyError, IndexError, ValueError) as e:
        log.warning("   PubChem parse error for '%s': %s", compound_name, e)
    except requests.exceptions.RequestException as e:
        log.warning("   Network error while querying PubChem: %s", e)
    return None


# ======================================================================
# Section 5 -- Core Inference Logic
# ======================================================================

def _align_features(raw: dict, feature_cols: list) -> pd.DataFrame:
    """
    Converts a raw feature dictionary into a single-row DataFrame whose
    columns exactly match the training feature layout. Missing columns are
    filled with 0 (safe default for binary fingerprint bits).

    Parameters
    ----------
    raw          : dict       -- Feature name to value mapping from featurization.
    feature_cols : list[str]  -- Ordered column list from training (feature_cols.pkl).

    Returns
    -------
    pd.DataFrame of shape (1, len(feature_cols)).
    """
    row = pd.DataFrame([raw])
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0
    return row[feature_cols]


def predict_flash_point(
    input_value: str,
    model,
    feature_cols: list,
    scaler=None,
    is_smiles: bool = False
) -> dict:
    """
    End-to-end inference for a single compound: resolves the input,
    featurizes it, runs the applicability domain check, aligns features,
    and returns a prediction dictionary.

    Parameters
    ----------
    input_value  : str               -- Compound name or SMILES string.
    model        : sklearn estimator -- Loaded predictive pipeline.
    feature_cols : list[str]         -- Training feature column names.
    scaler       : StandardScaler | None -- External scaler (or None if embedded).
    is_smiles    : bool              -- If True, treat input_value directly as SMILES.

    Returns
    -------
    dict with keys:
        input, smiles, predicted_fp, in_applicability_domain,
        domain_warnings, status, timestamp
    """
    result = {
        "input"                  : input_value,
        "smiles"                 : None,
        "predicted_fp"           : None,
        "in_applicability_domain": None,
        "domain_warnings"        : [],
        "status"                 : None,
        "timestamp"              : datetime.now().isoformat(),
    }

    # Step 1: Obtain SMILES
    if is_smiles:
        result["smiles"] = input_value
    else:
        result["smiles"] = resolve_name_to_smiles(input_value)

    if result["smiles"] is None:
        result["status"] = "resolution_error"
        return result

    # Step 2: Featurize SMILES into descriptor + fingerprint vector
    raw_features, mol_obj = smiles_to_features(result["smiles"])
    if raw_features is None:
        result["status"] = "smiles_error"
        return result

    # Step 3: Applicability Domain Evaluation
    # evaluate_applicability_domain returns a third flag (is_inorganic) that
    # acts as a hard gate: inorganic compounds are out of scope and cannot be
    # predicted regardless of any other domain criteria.
    in_domain, ad_warnings, is_inorganic = evaluate_applicability_domain(mol_obj, raw_features)
    result["in_applicability_domain"] = in_domain
    result["domain_warnings"]         = ad_warnings

    if is_inorganic:
        # Hard gate: block prediction. The caller (run_interactive_mode) is
        # responsible for displaying the user-facing message.
        result["status"] = "out_of_scope"
        return result

    # Step 4: Align feature vector to training schema
    X = _align_features(raw_features, feature_cols)

    # Step 5: Apply external scaler (bare estimators only).
    # When scaler is None it means either the model is a Pipeline with
    # preprocessing embedded, or no scaler artifact was found on disk.
    if scaler is not None:
        expected_n = getattr(scaler, "n_features_in_", None)
        if expected_n is not None and expected_n != X.shape[1]:
            log.warning(
                "   Scaler expects %d features but input has %d. "
                "Skipping external scaling to avoid corrupting the feature vector.",
                expected_n, X.shape[1]
            )
        else:
            try:
                X = pd.DataFrame(scaler.transform(X), columns=feature_cols)
            except Exception as e:
                log.warning("   External scaler transform failed: %s. Proceeding unscaled.", e)

    # Step 6: Predict
    result["predicted_fp"] = round(float(model.predict(X)[0]), 2)
    result["status"]        = "success"
    return result


# ======================================================================
# Section 6 -- Interactive Mode
# ======================================================================

def run_interactive_mode(model, feature_cols: list, scaler=None) -> None:
    """
    Launches an interactive REPL loop where the user can type compound names
    or SMILES strings one at a time and receive flash point predictions with
    applicability domain verification.
    Type 'exit' or 'quit' to terminate the session.

    Parameters
    ----------
    model        : estimator     -- Loaded model pipeline.
    feature_cols : list[str]     -- Training feature columns.
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
            status_msg = {
                "resolution_error": "compound name not resolved by PubChem",
                "smiles_error"    : "SMILES string could not be parsed by RDKit",
            }.get(result["status"], result["status"])
            print(f"  ! Prediction failed for '{result['input']}' -- {status_msg}")

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
                "input"       : r["input"],
                "predicted_fp": r["predicted_fp"],
                "in_domain"   : r.get("in_applicability_domain"),
                "status"      : r["status"],
                "warnings"    : warn_text,
            })
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))
        print("=" * 68 + "\n")

    # Restore log level so any post-session code can log normally.
    logging.getLogger().setLevel(logging.INFO)


# ======================================================================
# Script Entry Point
# ======================================================================

def main() -> None:
    """
    Main dispatcher -- loads model artifacts and launches the interactive
    prediction session.
    """
    log.info("=" * 70)
    log.info(" ChemPyQSPR -- Interactive Flash Point Inference Module")
    log.info(" Timestamp : %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 70)

    model, feature_cols, scaler = load_artifacts()
    run_interactive_mode(model, feature_cols, scaler)


if __name__ == "__main__":
    main()

