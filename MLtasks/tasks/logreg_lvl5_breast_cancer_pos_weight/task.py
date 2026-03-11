"""
Logistic Regression (Breast Cancer Pos Weight)
Binary logistic regression using breast cancer dataset with pos_weight.
- Data: sklearn.datasets.load_breast_cancer
- Standardize features
- Model: nn.Linear(d,1)
- Loss: BCEWithLogitsLoss(pos_weight=...)
- Metrics: confusion matrix + precision/recall/F1 on validation
- Validation: assert val recall > 0.90 (and/or improved vs unweighted baseline)
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Set seeds for reproducibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Get the computation device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_task_metadata():
    """Return task metadata."""
    return {
        "task_name": "logreg_lvl5_breast_cancer_pos_weight",
        "task_type": "binary_classification",
        "description": "Binary logistic regression on breast cancer dataset using BCEWithLogitsLoss(pos_weight=...)."
    }

def make_dataloaders(batch_size: int = 64, train_ratio: float = 0.8, seed: int = 42, device=None):
    """
    Load breast cancer dataset, standardize features, stratified 80/20 train/val.
    Returns loaders + numpy arrays (for baseline comparison if needed) + pos_weight tensor.
    """
    if device is None:
        device = get_device()

    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)  # 0/1 labels

    # Split first, then fit scaler on train only (avoid leakage)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=(1 - train_ratio), random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    # Compute pos_weight = (#negative / #positive) from TRAIN labels
    # In breast cancer dataset, class "1" is malignant? (doesn't matter; we treat y==1 as positive)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    # Guard (shouldn't happen here)
    pos_weight_value = float(n_neg / max(1, n_pos))
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    # Torch tensors
    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).float().unsqueeze(1).to(device)  # (N,1)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).float().unsqueeze(1).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_train, X_val, y_train, y_val, pos_weight, scaler

class LogisticRegressionModel(nn.Module):
    """
    IMPORTANT: return logits (no sigmoid here) because BCEWithLogitsLoss expects logits.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

def build_model(input_dim: int, device=None):
    """Build the logistic regression model."""
    if device is None:
        device = get_device()
    return LogisticRegressionModel(input_dim).to(device)

def train(model, train_loader, val_loader, device, pos_weight, lr=0.01, epochs=200, verbose_every=20):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running += loss.item()

        avg_train = running / len(train_loader)
        history["train_loss"].append(float(avg_train))

        # val loss
        model.eval()
        v_running = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                logits = model(Xb.to(device))
                v_running += criterion(logits, yb.to(device)).item()
        avg_val = v_running / len(val_loader)
        history["val_loss"].append(float(avg_val))

        if verbose_every and epoch % verbose_every == 0:
            print(f"Epoch [{epoch}/{epochs}], Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")

    return history

def predict_probs(model, X_tensor):
    """
    X_tensor: torch tensor on device
    Returns probabilities in numpy.
    """
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits)
    return probs.detach().cpu().numpy().reshape(-1)

def evaluate(model, data_loader, device):
    """
    Returns metrics on provided loader.
    Uses threshold 0.5.
    """
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for Xb, yb in data_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            logits = model(Xb)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.detach().cpu().numpy().reshape(-1).tolist())
            all_targets.extend(yb.detach().cpu().numpy().reshape(-1).tolist())

    all_probs = np.array(all_probs, dtype=np.float32)
    all_targets = np.array(all_targets, dtype=np.int64)
    preds = (all_probs >= 0.5).astype(np.int64)

    cm = confusion_matrix(all_targets, preds).tolist()
    precision = float(precision_score(all_targets, preds, zero_division=0))
    recall = float(recall_score(all_targets, preds, zero_division=0))
    f1 = float(f1_score(all_targets, preds, zero_division=0))
    acc = float(accuracy_score(all_targets, preds))

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }

def save_artifacts(model, metadata, train_metrics, val_metrics, history, pos_weight_value):
    """Save model, plots, and other artifacts."""
    payload = {
        "metadata": metadata,
        "train": train_metrics,
        "validation": val_metrics,
        "history": history,
        "pos_weight": float(pos_weight_value),
        "learned_params": {
            "weight_norm": float(torch.norm(model.linear.weight.detach()).cpu().item()),
            "bias": float(model.linear.bias.detach().cpu().item()),
        },
    }
    metrics_path = os.path.join(OUTPUT_DIR, "logreg_lvl5_breast_cancer_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)

    model_path = os.path.join(OUTPUT_DIR, "logreg_lvl5_breast_cancer_model.pt")
    torch.save(model.state_dict(), model_path)

    print(f"Artifacts saved to {OUTPUT_DIR}")

def main():
    """Main function to run the complete training and evaluation pipeline."""
    print("=" * 60)
    print("Logistic Regression (Breast Cancer Pos Weight)")
    print("=" * 60)

    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    metadata = get_task_metadata()

    print("\nCreating dataloaders...")
    train_loader, val_loader, X_train, X_val, y_train, y_val, pos_weight, _ = make_dataloaders(
        batch_size=64, train_ratio=0.8, seed=42, device=device
    )
    print(f"Train samples: {len(y_train)}, Val samples: {len(y_val)}")
    print(f"Train class counts [0,1]: {np.bincount(y_train)}")
    print(f"pos_weight (neg/pos): {pos_weight.item():.4f}")

    print("\nBuilding model...")
    model = build_model(input_dim=X_train.shape[1], device=device)
    print(f"Model: {model}")

    print("\nTraining model...")
    history = train(
        model,
        train_loader,
        val_loader,
        device=device,
        pos_weight=pos_weight,
        lr=0.01,
        epochs=200,
        verbose_every=20
    )

    print("\nEvaluating model...")
    train_metrics = evaluate(model, train_loader, device=device)
    val_metrics = evaluate(model, val_loader, device=device)

    print("\nMetrics:")
    print(f"  Train - Acc: {train_metrics['accuracy']:.4f}, Prec: {train_metrics['precision']:.4f}, Rec: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}")
    print(f"  Val   - Acc: {val_metrics['accuracy']:.4f}, Prec: {val_metrics['precision']:.4f}, Rec: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
    print(f"  Val Confusion Matrix: {val_metrics['confusion_matrix']}")

    print("\nSaving artifacts...")
    save_artifacts(model, metadata, train_metrics, val_metrics, history, pos_weight.item())

    # Quality checks (per your JSON requirement)
    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)

    passed = True

    recall_ok = val_metrics["recall"] > 0.90
    print(("✓" if recall_ok else "✗") + f" Val Recall > 0.90: {val_metrics['recall']:.4f}")
    passed = passed and recall_ok

    if passed:
        print("PASS: All quality checks passed!")
        return 0
    else:
        print("FAIL: Some quality checks failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
