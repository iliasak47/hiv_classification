import argparse
import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import Descriptors

def compute_descriptors(smiles):
    """
    Computes a list of basic RDKit descriptors from a SMILES string.
    Returns None if the SMILES is invalid.
    """
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MolWt": Descriptors.MolWt(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "NumAromaticRings": Descriptors.NumAromaticRings(mol),
        "LogP": Descriptors.MolLogP(mol)
    }

def featurize_dataset(input_path):
    """
    Loads a CSV file, computes descriptors from SMILES,
    and returns a new DataFrame with features added.
    """
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)

    # Apply descriptor calculation
    descriptors = df['canonical_smiles'].apply(compute_descriptors)

    # Drop invalid rows
    valid_idx = descriptors.dropna().index
    df_valid = df.loc[valid_idx].reset_index(drop=True)
    descriptors_df = pd.DataFrame(list(descriptors.dropna())).reset_index(drop=True)

    # Merge features into the original data
    df_featurized = pd.concat([df_valid, descriptors_df], axis=1)

    print(f"{len(df_featurized)} valid molecules with descriptors extracted.")
    return df_featurized


def save_featurized_data(df, output_path):
    """
    Saves the featurized DataFrame to CSV.
    Creates the output directory if it doesn't exist.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved featurized data to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Featurize molecules from SMILES using RDKit")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file with canonical_smiles column")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output featurized CSV")
    args = parser.parse_args()

    df_out = featurize_dataset(args.input)
    save_featurized_data(df_out, args.output)
