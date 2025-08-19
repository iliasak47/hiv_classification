import sys, subprocess, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "MolWt","TPSA","NumRotatableBonds","NumHDonors",
    "NumHAcceptors","NumAromaticRings","LogP",
]

SCRIPT = Path("src/inference/predict.py")


def make_ts_model(tmp_path: Path, prob: float) -> Path:
    """Create a TorchScript model that outputs a constant logit = logit(prob)."""
    logit = math.log(prob / (1.0 - prob))

    class M(torch.nn.Module):
        def __init__(self, l: float):
            super().__init__()
            self.l = torch.tensor(float(l))
        def forward(self, x):  # x: (N, 7)
            n = x.shape[0]
            return torch.full((n,), self.l)

    m = torch.jit.script(M(logit))
    p = tmp_path / "model_ts.pt"
    m.save(str(p))
    return p


def write_artifacts(tmp_path: Path, df_fit: pd.DataFrame) -> tuple[Path, Path]:
    scaler = StandardScaler().fit(df_fit.values)
    scaler_path = tmp_path / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    feats_path = tmp_path / "feature_list.joblib"
    joblib.dump(FEATURES, feats_path)
    return scaler_path, feats_path


def run_predict(input_csv: Path, output_csv: Path, model_path: Path, scaler_path: Path, feats_path: Path, threshold: float | None = None, id_col: str | None = None):
    cmd = [
        sys.executable, str(SCRIPT),
        "--input", str(input_csv),
        "--output", str(output_csv),
        "--model", str(model_path),
        "--scaler", str(scaler_path),
        "--features", str(feats_path),
    ]
    if threshold is not None:
        cmd += ["--threshold", str(threshold)]
    if id_col is not None:
        cmd += ["--id-col", id_col]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_cli_happy_path(tmp_path: Path):
    df = pd.DataFrame({f: np.random.rand(10) + 1 for f in FEATURES})
    df["chembl_id"] = [f"CHEMBL{i}" for i in range(len(df))]
    csv_in = tmp_path / "in.csv"; df.to_csv(csv_in, index=False)

    scaler_path, feats_path = write_artifacts(tmp_path, df[FEATURES])
    model_path = make_ts_model(tmp_path, prob=0.9)  # prob>threshold

    out_csv = tmp_path / "out.csv"
    res = run_predict(csv_in, out_csv, model_path, scaler_path, feats_path, threshold=0.35, id_col="chembl_id")
    assert res.returncode == 0, res.stderr

    out = pd.read_csv(out_csv)
    assert len(out) == len(df)
    assert {"prob_active","pred_active","chembl_id"}.issubset(out.columns)
    assert out["pred_active"].eq(1).all()


def test_threshold_effect(tmp_path: Path):
    df = pd.DataFrame({f: np.random.rand(6) + 1 for f in FEATURES})
    csv_in = tmp_path / "in.csv"; df.to_csv(csv_in, index=False)

    scaler_path, feats_path = write_artifacts(tmp_path, df[FEATURES])
    model_path = make_ts_model(tmp_path, prob=0.4)  # below 0.5

    out_low = tmp_path / "out_low.csv"
    res1 = run_predict(csv_in, out_low, model_path, scaler_path, feats_path, threshold=0.5)
    assert res1.returncode == 0
    o1 = pd.read_csv(out_low)
    assert o1["pred_active"].eq(0).all()

    out_high = tmp_path / "out_high.csv"
    res2 = run_predict(csv_in, out_high, model_path, scaler_path, feats_path, threshold=0.35)
    assert res2.returncode == 0
    o2 = pd.read_csv(out_high)
    assert o2["pred_active"].eq(1).all()


def test_missing_feature_error(tmp_path: Path):
    partial_feats = FEATURES.copy(); partial_feats.remove("TPSA")
    df = pd.DataFrame({f: np.random.rand(5) + 1 for f in partial_feats})
    csv_in = tmp_path / "in.csv"; df.to_csv(csv_in, index=False)

    # Artifacts still expect full feature set
    scaler_path, feats_path = write_artifacts(tmp_path, pd.DataFrame({f: np.random.rand(8)+1 for f in FEATURES}))
    model_path = make_ts_model(tmp_path, prob=0.9)

    out_csv = tmp_path / "out.csv"
    res = run_predict(csv_in, out_csv, model_path, scaler_path, feats_path)
    assert res.returncode != 0
    assert "Missing required feature columns" in (res.stderr + res.stdout)


def test_nan_imputation(tmp_path: Path):
    clean = pd.DataFrame({f: np.random.rand(8) + 1 for f in FEATURES})
    scaler_path, feats_path = write_artifacts(tmp_path, clean)

    df = clean.copy()
    df.loc[0, "MolWt"] = np.nan
    df.loc[3, "NumHDonors"] = np.nan
    csv_in = tmp_path / "in.csv"; df.to_csv(csv_in, index=False)

    model_path = make_ts_model(tmp_path, prob=0.6)
    out_csv = tmp_path / "out.csv"
    res = run_predict(csv_in, out_csv, model_path, scaler_path, feats_path, threshold=0.35)
    assert res.returncode == 0, res.stderr
    out = pd.read_csv(out_csv)
    assert len(out) == len(df)
