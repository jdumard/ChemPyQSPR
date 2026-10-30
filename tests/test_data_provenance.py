
from pathlib import Path
import sys

import pandas as pd

# Import the collection module
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import collect_flash_points as collector


def test_data_provenance(monkeypatch, tmp_path, caplog):
    # Create controlled source data
    raw_csv = tmp_path / "source.csv"

    pd.DataFrame([
        {
            "compound": "ethanol",
            "smiles": "CCO",
            "flashpoint": 283.15,
            "source": "lab_A"
        },
        {
            "compound": "ethanol",
            "smiles": "CCO",
            "flashpoint": 285.15,
            "source": "lab_B"
        },
        {
            "compound": "acetone",
            "smiles": "CC(=O)C",
            "flashpoint": 250.15,
            "source": "lab_C"
        }
    ]).to_csv(raw_csv, index=False)

    # Replace external data dependencies
    monkeypatch.setattr(
        collector,
        "ensure_curated_file_exists",
        lambda: True
    )
    monkeypatch.setattr(
        collector,
        "RAW_CSV",
        str(raw_csv)
    )
    monkeypatch.setattr(
        collector,
        "get_pubchem_records",
        lambda: [{
            "name": "ethanol",
            "smiles": "CCO",
            "flash_point": 14.0,
            "source_dataset": "PubChem",
            "source_record_id": "CID_test",
            "source_reference": None
        }]
    )

    # Redirect outputs to a temporary directory
    monkeypatch.setattr(
        collector, "OUTPUT_DIR", str(tmp_path)
    )
    monkeypatch.setattr(
        collector,
        "OUTPUT_CSV",
        str(tmp_path / "flash_point_database.csv")
    )
    monkeypatch.setattr(
        collector,
        "RECORDS_CSV",
        str(tmp_path / "records.csv")
    )

    # Enable INFO-level log capture
    caplog.set_level("INFO", logger=collector.log.name)

    # Run the complete collection workflow
    collector.main()

    # Verify pipeline summary messages
    assert "Saved record-level dataset" in caplog.text
    assert "Saved consolidated dataset" in caplog.text
    assert "Total input records" in caplog.text
    assert "Unique compounds" in caplog.text
    assert "Records aggregated" in caplog.text

    records = pd.read_csv(tmp_path / "records.csv")
    final = pd.read_csv(
        tmp_path / "flash_point_database.csv"
    )

    # Validate record preservation
    assert len(records) == 4
    assert records["source_record_id"].notna().all()

    assert set(records["source_dataset"]) == {
        "CHEMINFO", "PubChem"
    }

    assert set(
        records["source_reference"].dropna()
    ) == {"lab_A", "lab_B", "lab_C"}

    # Validate backward compatibility
    assert len(final) == 2
    assert list(final.columns) == [
        "name", "smiles", "flash_point"
    ]

    # Ethanol median: 10, 12 and 14 degrees Celsius
    ethanol_fp = final.loc[
        final["smiles"] == "CCO",
        "flash_point"
    ].iloc[0]

    assert abs(ethanol_fp - 12.0) < 1e-6

    # Acetone: 250.15 K = -23 degrees Celsius
    acetone_fp = final.loc[
        final["smiles"] == "CC(C)=O",
        "flash_point"
    ].iloc[0]

    assert abs(acetone_fp + 23.0) < 1e-6
