# HIV Classification Project

## Overview
This project aims to classify compounds as **Active** or **Inactive** against the HIV-1 Reverse Transcriptase enzyme using cheminformatics features.  
We leverage both **Apache Spark (PySpark)** for scalable data processing and **PyTorch** for building a neural network model.  
The work includes **data processing**, **model training**, **hyperparameter optimization**, and **model deployment preparation**.

---

## Project Structure
```
hiv_classification/
│
├── data/                 # Raw and processed datasets
├── notebooks/            # Exploration and experimentation
├── src/
│   ├── data/             # Data loading and processing scripts
│   ├── features/         # Feature engineering
│   ├── models/           # Training and evaluation scripts
│
└── README.md             # Project documentation
```

---

## Features
For each molecule, the following features are computed using RDKit:
- **MolWt**: Molecular weight
- **TPSA**: Topological Polar Surface Area
- **NumRotatableBonds**: Number of rotatable bonds
- **NumHDonors**: Number of hydrogen bond donors
- **NumHAcceptors**: Number of hydrogen bond acceptors
- **NumAromaticRings**: Number of aromatic rings
- **LogP**: Partition coefficient (log P)

---

## Data Processing
- Data is fetched from **ChEMBL** and stored as CSV/Parquet.
- Features are computed using RDKit.
- Label **`active`** is created based on `standard_value < 1000` nM.

---

## Models
We implemented two main models:
1. **Random Forest Classifier** (PySpark MLlib)
2. **Multi-Layer Perceptron (MLP)** (PyTorch)

### Performance (Test Set)
| Model         | Threshold | Accuracy | Precision | Recall | F1-score | ROC AUC |
|---------------|-----------|----------|-----------|--------|----------|---------|
| RF (Spark)    | 0.50      | 0.712    | 0.705     | 0.718  | 0.711    | 0.726   |
| MLP (PyTorch) | 0.50      | 0.754    | 0.743     | 0.787  | 0.764    | 0.835   |

---

## Training
### Spark Model
- Hyperparameter tuning with `ParamGridBuilder` and `CrossValidator`
- Metrics logged with MLflow

### PyTorch Model
- Architecture: 7 → 32 → 16 → 1 (ReLU activations)
- Loss: `BCEWithLogitsLoss`
- Optimizer: Adam
- StandardScaler applied to features
- Metrics: Accuracy, Precision, Recall, F1, ROC AUC
- Best threshold chosen based on max F1-score
- All artifacts logged with MLflow



## Next Steps
- Add unit tests
- Implement continuous training pipeline
- Add monitoring for data drift and model performance

---

## Author
This project was developed as part of a machine learning portfolio to showcase **end-to-end ML workflow skills** including:
- Data engineering with Spark
- Feature engineering with RDKit
- Model development in PyTorch
- Experiment tracking with MLflow
- Reproducibility and deployment preparation
