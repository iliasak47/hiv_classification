from pathlib import Path
import importlib.util
import types
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# --- Load predict.py as a module from file path ---
PREDICT_PATH = Path(__file__).resolve().parents[1] / "src/inference/predict.py"
spec = importlib.util.spec_from_file_location("predict_mod", PREDICT_PATH)
predict = importlib.util.module_from_spec(spec)  # type: ignore
assert spec and spec.loader
spec.loader.exec_module(predict)  # type: ignore

FEATURES = [
    "MolWt","TPSA","NumRotatableBonds","NumHDonors",
    "NumHAcceptors","NumAromaticRings","LogP",
]


def test_load_features_fallback_when_missing(tmp_path: Path):
    feats = predict.load_features(str(tmp_path / "missing.joblib"))
    assert isinstance(feats, list)
    assert len(feats) == 7


def test_load_features_joblib_list(tmp_path: Path):
    p = tmp_path / "feature_list.joblib"
    joblib.dump(FEATURES, p)
    feats = predict.load_features(str(p))
    assert feats == FEATURES


def test_ensure_features_ok():
    df = pd.DataFrame({f: np.ones(3) for f in FEATURES})
    out = predict.ensure_features(df, FEATURES)
    assert list(out.columns) == FEATURES


def test_ensure_features_missing_column():
    partial = FEATURES[:-1]
    df = pd.DataFrame({f: np.ones(3) for f in partial})
    try:
        predict.ensure_features(df, FEATURES)
        assert False, "Expected SystemExit due to missing feature"
    except SystemExit as e:
        assert "Missing required feature columns" in str(e)


def test_impute_with_scaler_mean_imputes_nans():
    # Fit scaler with known means
    train = pd.DataFrame({
        "MolWt": [10, 10],
        "TPSA": [20, 20],
        "NumRotatableBonds": [30, 30],
        "NumHDonors": [40, 40],
        "NumHAcceptors": [50, 50],
        "NumAromaticRings": [60, 60],
        "LogP": [70, 70],
    })
    scaler = StandardScaler().fit(train.values)

    X = train.copy()
    X.loc[0, "MolWt"] = np.nan
    X.loc[1, "NumHDonors"] = np.nan

    out = predict.impute_with_scaler_mean(X.copy(), scaler)
    assert not out.isna().any().any()
    assert out.loc[0, "MolWt"] == 10
    assert out.loc[1, "NumHDonors"] == 40


def test_prompt_for_input_path_returns_given(tmp_path: Path):
    p = tmp_path / "in.csv"
    p.write_text("MolWt\n1\n")
    out = predict.prompt_for_input_path(str(p))
    assert out == str(p)
