"""
collect_flash_points.py
=======================
Consolidated collector combining the CHEMINFO Molecule-Features dataset
and PubChem API queries. Performs chemical sanitization with RDKit,
unit conversions, canonical SMILES deduplication, and exports a clean CSV dataset
with columns: name, smiles, flash_point.
"""

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
import os
import re
import time
import json
import logging
import urllib.request
import urllib.parse
import pandas as pd
from rdkit import Chem

# ----------------------------------------------------------------------
# Global Paths and Configuration
# ----------------------------------------------------------------------
OUTPUT_DIR   = os.path.join("data", "processed")
OUTPUT_CSV   = os.path.join(OUTPUT_DIR, "flash_point_database.csv")
RAW_DATA_DIR = os.path.join("data", "raw")
RAW_CSV      = os.path.join(RAW_DATA_DIR, "molecule_features_flashpoint.csv")
DATA_URL     = "https://raw.githubusercontent.com/cheminfo/molecule-features/main/data/flashpoint/data.csv"

PUBCHEM_REST_BASE  = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_VIEW_BASE  = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
PUBCHEM_HEADERS    = {"User-Agent": "ChemPyQSPR_AcademicResearch/3.2 (chempyqspr@research.edu)"}
PUBCHEM_BATCH_SIZE = 50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Chemical Sanitization and Validation
# ----------------------------------------------------------------------
def is_organic(mol: Chem.Mol) -> bool:
    """Validates if the RDKit molecule is a single-component organic compound."""
    if mol is None or "." in Chem.MolToSmiles(mol):
        return False
    return any(atom.GetSymbol() == "C" for atom in mol.GetAtoms())


def convert_to_celsius(value: float, unit: str = "C") -> float:
    """Converts Fahrenheit or Kelvin to Celsius."""
    if "F" in unit.upper():
        return round((value - 32.0) * 5.0 / 9.0, 2)
    elif "K" in unit.upper():
        return round(value - 273.15, 2)
    return round(value, 2)


# ----------------------------------------------------------------------
# Source 1: CHEMINFO Molecule-Features GitHub Dataset Ingestion
# ----------------------------------------------------------------------
def ensure_curated_file_exists() -> bool:
    """
    Attempts automatic download of the CHEMINFO Molecule-Features dataset.
    If automatic download fails, prompts the user with manual download instructions.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_CSV) and os.path.getsize(RAW_CSV) > 0:
        return True

    log.info("[MoleculeFeatures] Attempting automatic download from %s...", DATA_URL)
    try:
        urllib.request.urlretrieve(DATA_URL, RAW_CSV)
        if os.path.exists(RAW_CSV) and os.path.getsize(RAW_CSV) > 0:
            log.info("[MoleculeFeatures] Automatic download successful!")
            return True
    except Exception as exc:
        log.warning("[MoleculeFeatures] Automatic download failed: %s", exc)

    # Interactive manual download prompt fallback
    while not os.path.exists(RAW_CSV) or os.path.getsize(RAW_CSV) == 0:
        print("\n" + "=" * 75)
        print("ACTION REQUIRED: Automatic download failed or file missing.")
        print("=" * 75)
        print("Please download the CHEMINFO Molecule-Features dataset manually:")
        print(f"1. Open the URL in your browser:\n   {DATA_URL}")
        print("2. Save the dataset file as 'molecule_features_flashpoint.csv'.")
        print(f"3. Place the file at:\n   {os.path.abspath(RAW_CSV)}")
        print("=" * 75)

        user_choice = input("\nHave you saved the file at the required location? (y/n/skip): ").strip().lower()

        if user_choice == "y":
            if os.path.exists(RAW_CSV) and os.path.getsize(RAW_CSV) > 0:
                print("Dataset detected successfully! Resuming pipeline...\n")
                return True
            else:
                print(f"File not found at '{RAW_CSV}'. Please verify path and filename.")
        elif user_choice == "skip":
            log.warning("Skipping Molecule-Features dataset ingestion upon user request.")
            return False
        else:
            print("Operation paused. Please save the file and confirm.")

    return True


def get_molecule_features_records() -> list[dict]:
    """Extracts and validates organic records from the Molecule-Features dataset."""
    if not ensure_curated_file_exists():
        return []

    log.info("[MoleculeFeatures] Reading dataset from: %s", RAW_CSV)
    try:
        df = pd.read_csv(RAW_CSV)
    except Exception as exc:
        log.error("[MoleculeFeatures] Failed to read CSV file: %s", exc)
        return []

    # Exclude non-organic flags if present
    for col in ["is_metallic", "is_silicon", "is_tin"]:
        if col in df.columns:
            df = df[df[col] == 0]

    # Vectorized Kelvin -> Celsius conversion and temperature bounds filtering
    df["flash_point_C"] = (df["flashpoint"] - 273.15).round(2)
    df = df[(df["flash_point_C"] >= -150.0) & (df["flash_point_C"] <= 500.0)]

    records = []
    for _, row in df.iterrows():
        smi = str(row.get("smiles", "")).strip()
        if not smi:
            continue

        mol = Chem.MolFromSmiles(smi)
        if not is_organic(mol):
            continue

        can_smi = Chem.MolToSmiles(mol, canonical=True)
        name = str(row.get("compound", "")).strip() if pd.notna(row.get("compound")) else f"Compound_{can_smi}"

        records.append({
            "name": name,
            "smiles": can_smi,
            "flash_point": float(row["flash_point_C"])
        })

    log.info("[MoleculeFeatures] Extracted %d valid organic records.", len(records))
    return records


# ----------------------------------------------------------------------
# Source 2: PubChem Global Annotations Collector
# ----------------------------------------------------------------------
def _parse_annotation_fp(entry: dict) -> float | None:
    """Extracts and normalizes experimental flash point from a PubChem annotation entry."""
    for datum in entry.get("Data", []):
        val = datum.get("Value", {})

        # Direct numeric value
        if "Number" in val:
            num = val["Number"]
            num_val = num[0] if isinstance(num, list) and num else num
            unit = val.get("Unit", "C")
            try:
                return convert_to_celsius(float(num_val), str(unit))
            except (TypeError, ValueError):
                pass

        # Free-text parsing (°C / °F)
        for mk in val.get("StringWithMarkup", []):
            text = mk.get("String", "")
            m_c = re.search(r'(-?\d+(?:\.\d+)?)\s*(?:°|\bdeg\b)?\s*C', text, re.I)
            if m_c:
                return float(m_c.group(1))
            m_f = re.search(r'(-?\d+(?:\.\d+)?)\s*(?:°|\bdeg\b)?\s*F', text, re.I)
            if m_f:
                return convert_to_celsius(float(m_f.group(1)), "F")
    return None


def get_pubchem_records() -> list[dict]:
    """Queries global Flash Point annotations from PubChem API and extracts properties."""
    log.info("[PubChem] Querying global Flash Point annotations from PubChem API...")
    base_annot_url = f"{PUBCHEM_VIEW_BASE}/annotations/heading/JSON"
    params = {"heading_type": "Compound", "heading": "Flash Point"}

    try:
        first_url = f"{base_annot_url}?{urllib.parse.urlencode({**params, 'page': 1})}"
        req = urllib.request.Request(first_url, headers=PUBCHEM_HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        log.error("[PubChem] Failed to connect to PubChem API: %s", exc)
        return []

    total_pages = data.get("Annotations", {}).get("TotalPages", 1)
    log.info("[PubChem] Total annotation pages to scan: %d", total_pages)

    cid_fp: dict[int, float] = {}

    # 1. Map CID -> Flash Point across pages
    for page in range(1, total_pages + 1):
        if page > 1:
            try:
                page_url = f"{base_annot_url}?{urllib.parse.urlencode({**params, 'page': page})}"
                req = urllib.request.Request(page_url, headers=PUBCHEM_HEADERS)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                time.sleep(0.15)
            except Exception:
                continue

        for entry in data.get("Annotations", {}).get("Annotation", []):
            cids = entry.get("LinkedRecords", {}).get("CID", [])
            if cids:
                cid = int(cids[0])
                fp = _parse_annotation_fp(entry)
                if fp is not None and cid not in cid_fp:
                    cid_fp[cid] = fp

    log.info("[PubChem] Unique CIDs with experimental Flash Point: %d", len(cid_fp))
    if not cid_fp:
        return []

    # 2. Batch property lookup via POST
    all_cids = list(cid_fp.keys())
    chunks = [all_cids[i : i + PUBCHEM_BATCH_SIZE] for i in range(0, len(all_cids), PUBCHEM_BATCH_SIZE)]
    records: list[dict] = []
    prop_url = f"{PUBCHEM_REST_BASE}/compound/cid/property/CanonicalSMILES,ConnectivitySMILES,IsomericSMILES,IUPACName,Title/JSON"

    for chunk in chunks:
        post_data = urllib.parse.urlencode({"cid": ",".join(map(str, chunk))}).encode("utf-8")
        try:
            req = urllib.request.Request(prop_url, data=post_data, headers=PUBCHEM_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                prop_data = json.loads(resp.read().decode())
                props = prop_data.get("PropertyTable", {}).get("Properties", [])
            time.sleep(0.2)
        except Exception:
            continue

        for p in props:
            cid = p.get("CID")
            smi = p.get("ConnectivitySMILES") or p.get("CanonicalSMILES") or p.get("IsomericSMILES") or p.get("SMILES", "")
            if not smi or cid not in cid_fp:
                continue

            mol = Chem.MolFromSmiles(smi)
            if not is_organic(mol):
                continue

            fp_c = cid_fp[cid]
            if fp_c < -150.0 or fp_c > 500.0:
                continue

            can_smi = Chem.MolToSmiles(mol, canonical=True)
            name = p.get("Title") or p.get("IUPACName") or f"CID_{cid}"

            records.append({
                "name": str(name),
                "smiles": can_smi,
                "flash_point": fp_c
            })

    log.info("[PubChem] Extracted %d valid organic records.", len(records))
    return records


# ----------------------------------------------------------------------
# Pipeline Orchestration and Export
# ----------------------------------------------------------------------
def main():
    log.info("=" * 65)
    log.info("STARTING FLASH POINT INGESTION AND DEDUPLICATION PIPELINE")
    log.info("=" * 65)

    all_records = get_molecule_features_records() + get_pubchem_records()
    if not all_records:
        log.error("No valid records collected from any source.")
        return

    df_raw = pd.DataFrame(all_records)
    total_raw = len(df_raw)

    # Deduplication aggregated by canonical SMILES using median flash point
    df_final = (
        df_raw.groupby("smiles", as_index=False)
        .agg({
            "name": "first",
            "flash_point": "median"
        })
        .sort_values("flash_point")
        .reset_index(drop=True)
    )

    df_final = df_final[["name", "smiles", "flash_point"]]

    # CSV Export
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    log.info("=" * 65)
    log.info("PIPELINE COMPLETED SUCCESSFULLY")
    log.info("Saved output file   : %s", OUTPUT_CSV)
    log.info("Total compounds (N) : %d", len(df_final))
    log.info("Duplicates pruned   : %d", total_raw - len(df_final))
    log.info("Flash Point range   : %.2f deg C to %.2f deg C", df_final["flash_point"].min(), df_final["flash_point"].max())
    log.info("=" * 65)

    print("\nDataset Preview (First 5 records):")
    print(df_final.head().to_string(index=False))


if __name__ == "__main__":
    main()