import os
import joblib
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ---------------------------
# 1. Configs &  Reproducibility
# ---------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = "data/processed/hiv_ic50_featurized.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------
# 2. Prepare Data
# ---------------------------
print(" Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Create target column "active"
df["active"] = (df["standard_value"] < 1000).astype(int)  # adjust threshold if needed

features = ["MolWt", "TPSA", "NumRotatableBonds", "NumHDonors",
            "NumHAcceptors", "NumAromaticRings", "LogP"]
X = df[features].values
y = df["active"].values.reshape(-1, 1).astype(np.float32)

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
joblib.dump(features, os.path.join(MODEL_DIR, "feature_list.joblib"))

# Torch datasets
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ---------------------------
# 3. Define Model
# ---------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(7, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()

# ---------------------------
# 4. Loss & Optimizer
# ---------------------------
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ---------------------------
# 5. Training Loop
# ---------------------------
def train(model, loader):
    model.train()
    running_loss = 0.0
    for features, labels in loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

# ---------------------------
# 6. Evaluation Function
# ---------------------------
def evaluate(model, loader, threshold=0.35):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features)
            probs = torch.sigmoid(outputs)
            all_labels.extend(labels.cpu().numpy().ravel())
            all_probs.extend(probs.cpu().numpy().ravel())
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    preds = (all_probs >= threshold).astype(int)
    acc = accuracy_score(all_labels, preds)
    prec = precision_score(all_labels, preds)
    rec = recall_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    auc = roc_auc_score(all_labels, all_probs)
    return acc, prec, rec, f1, auc, all_probs, all_labels

# ---------------------------
# 7. MLflow Logging
# ---------------------------
mlflow.set_experiment("hiv_pytorch")
with mlflow.start_run():
    mlflow.log_params({
        "seed": SEED,
        "lr": 0.01,
        "batch_size": 32,
        "epochs": 50,
        "architecture": "7-32-16-1",
        "loss": "BCEWithLogitsLoss",
        "optimizer": "Adam"
    })

    for epoch in range(50):
        loss = train(model, train_loader)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    acc, prec, rec, f1, auc, probs, labels = evaluate(model, test_loader)
    print(f" Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, Recall: {rec:.4f}")
    mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc})

    # Save model & threshold
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "pytorch_mlp.pt"))
    mlflow.pytorch.log_model(model, "pytorch_mlp")
