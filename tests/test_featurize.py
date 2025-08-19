import os
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")  # skip if RDKit not installed

# Load featurize.py symbols
from src.data.featurize import compute_descriptors, featurize_dataset, save_featurized_data

FEATURE_COLS = [
    "MolWt",
    "TPSA",
    "NumRotatableBonds",
    "NumHDonors",
    "NumHAcceptors",
    "NumAromaticRings",
    "LogP",
]


def test_compute_descriptors_valid_and_invalid():
    d_valid = compute_descriptors("CCO")  # ethanol
    assert isinstance(d_valid, dict)
    assert set(FEATURE_COLS).issubset(d_valid.keys())
    assert all(isinstance(d_valid[k], (int, float, np.floating)) for k in FEATURE_COLS)

    assert compute_descriptors("INVALID") is None
    assert compute_descriptors(None) is None


def test_featurize_dataset_drops_invalid_and_adds_columns(tmp_path: Path):
    df_in = pd.DataFrame({
        "canonical_smiles": ["CCO", "c1ccccc1", "INVALID", None],
        "id": [1, 2, 3, 4],
    })
    csv_in = tmp_path / "raw.csv"
    df_in.to_csv(csv_in, index=False)

    out = featurize_dataset(str(csv_in))

    # Keep only valid rows (2 valid)
    assert len(out) == 2
    for c in FEATURE_COLS:
        assert c in out.columns
        assert out[c].notna().all()
        assert pd.api.types.is_numeric_dtype(out[c])


def test_save_featurized_data_writes_file(tmp_path: Path):
    df = pd.DataFrame({"canonical_smiles": ["CCO"], **{c: [1.0] for c in FEATURE_COLS}})
    out_path = tmp_path / "data" / "featurized.csv"
    save_featurized_data(df, str(out_path))
    assert out_path.exists()
    df2 = pd.read_csv(out_path)
    assert set(["canonical_smiles", *FEATURE_COLS]).issubset(df2.columns)
