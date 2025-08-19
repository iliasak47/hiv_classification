#!/usr/bin/env python3
import argparse, os, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd, torch

DEFAULT_MODEL = "models/pytorch_mlp_ts.pt"
DEFAULT_SCALER = "models/scaler.joblib"
DEFAULT_FEATURES = "models/feature_list.joblib"
DEFAULT_THRESHOLD = 0.35
DEFAULT_ID_COLS = ["chembl_id", "molecule_id", "compound_id", "id"]

def prompt_for_input_path(p):
    if p:
        return p
    try:
        path = input("Enter path to input CSV: ").strip().strip('"').strip("'")
    except EOFError:
        path = ""
    if not path or not os.path.exists(path):
        sys.exit("Error: please provide a valid path to the input CSV.")
    return path


def load_features(path):
    if not os.path.exists(path):
        return [
            "MolWt",
            "TPSA",
            "NumRotatableBonds",
            "NumHDonors",
            "NumHAcceptors",
            "NumAromaticRings",
            "LogP",
        ]
    data = joblib.load(path)
    if isinstance(data, dict) and "features" in data:
        return list(data["features"])  # type: ignore[return-value]
    if isinstance(data, (list, tuple)):
        return list(data)
    sys.exit("Invalid features artifact.")


def ensure_features(df, feats):
    missing = [f for f in feats if f not in df.columns]
    if missing:
        sys.exit(f"Missing required feature columns: {missing}")
    return df[feats].copy()


def impute_with_scaler_mean(X, scaler):
    if hasattr(scaler, "mean_") and len(getattr(scaler, "mean_")) == X.shape[1]:
        means = np.asarray(scaler.mean_)
        for i, c in enumerate(X.columns):
            if X[c].isna().any():
                X[c] = X[c].fillna(means[i])
        return X
    return X.fillna(X.median(numeric_only=True))


def parse_args():
    p = argparse.ArgumentParser(description="Predict HIV activity")
    p.add_argument("--input", default=None)
    p.add_argument("--output", default="predictions/preds.csv")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--scaler", default=DEFAULT_SCALER)
    p.add_argument("--features", default=DEFAULT_FEATURES)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--id-col", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    in_path = prompt_for_input_path(args.input)
    df = pd.read_csv(in_path)

    features = load_features(args.features)
    scaler = joblib.load(args.scaler)
    model = torch.jit.load(args.model, map_location="cpu"); model.eval()

    X_df = ensure_features(df, features)
    X_df = impute_with_scaler_mean(X_df, scaler)
    X = scaler.transform(X_df)

    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        if logits.ndim > 1:
            logits = logits.squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= float(args.threshold)).astype(int)

    out = pd.DataFrame({"prob_active": probs, "pred_active": preds})
    id_cols = [args.id_col] if args.id_col and args.id_col in df.columns else [c for c in DEFAULT_ID_COLS if c in df.columns]
    if id_cols:
        out = pd.concat([df[id_cols].reset_index(drop=True), out], axis=1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Saved {len(out)} predictions to '{args.output}'. Threshold={args.threshold}.")


if __name__ == "__main__":
    main()
