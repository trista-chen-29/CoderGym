"""
Logistic Regression (LBFGS Iris Softmax)
Softmax regression on Iris using LBFGS optimizer with closure.
- Data: sklearn.datasets.load_iris
- Standardize features (train-fit only)
- Model: nn.Linear(4,3) + CrossEntropyLoss
- Optimizer: torch.optim.LBFGS (closure)
- Metrics: val loss, accuracy, macro-F1
- Validation: assert val accuracy > 0.90 and macro-F1 > 0.90
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Ensure output directory exists
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_task_metadata():
    """Return task metadata."""
    return {
        "task_name": "logreg_lvl6_lbfgs_iris_softmax",
        "task_type": "classification",
        "num_classes": 3,
        "input_dim": 4,
        "description": "Softmax regression on Iris using LBFGS optimizer with closure.",
    }

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Get the appropriate device (CUDA or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_dataloaders(batch_size=64, train_ratio=0.8, seed=42, device=None):
    """
    Load Iris, standardize, stratified 80/20 split.
    Returns loaders + numpy arrays for reporting.
    """
    if device is None:
        device = get_device()

    data = load_iris()
    X = data.data.astype(np.float32)  # (150, 4)
    y = data.target.astype(np.int64)  # (150,)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=(1 - train_ratio), random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_train, X_val, y_train, y_val, scaler

class SoftmaxRegressionModel(nn.Module):
    """nn.Linear(4,3) -> logits"""

    def __init__(self, input_dim=4, num_classes=3):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)  # logits

def build_model(input_dim, num_classes, device):
    """Build and return the model."""
    model = SoftmaxRegressionModel(input_dim, num_classes).to(device)
    return model

def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the model and return metrics.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for Xb, yb in data_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            logits = model(Xb)
            loss = criterion(logits, yb)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(yb.cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(data_loader))
    acc = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average="macro")
    cm = confusion_matrix(all_targets, all_preds).tolist()

    return {
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "confusion_matrix": cm,
    }

def train_lbfgs(model, train_loader, criterion, device, max_iter=200, lr=1.0, verbose=True):
    """
    LBFGS is a full-batch-ish optimizer. We'll compute loss over ALL train batches in the closure.
    This is the key requirement for lvl6.
    """
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_iter,
        history_size=10,
        line_search_fn="strong_wolfe",
    )

    # LBFGS calls closure multiple times per step; closure must recompute loss + grads.
    def closure():
        optimizer.zero_grad()
        total_loss = 0.0

        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            logits = model(Xb)
            loss = criterion(logits, yb)
            total_loss += loss

        # average loss across batches
        total_loss = total_loss / max(1, len(train_loader))
        total_loss.backward()
        return total_loss

    model.train()
    final_loss = optimizer.step(closure)

    if verbose:
        # one more forward pass to show a stable number
        with torch.no_grad():
            loss_val = float(final_loss.item()) if hasattr(final_loss, "item") else float(final_loss)
        print(f"LBFGS finished. Final (reported) train loss: {loss_val:.6f}")

    return float(final_loss.item()) if hasattr(final_loss, "item") else float(final_loss)

def save_artifacts(model, metrics):
    """Save model artifacts and visualizations."""
    model_path = os.path.join(OUTPUT_DIR, "logreg_lvl6_lbfgs_iris_model.pt")
    torch.save(model.state_dict(), model_path)

    metrics_path = os.path.join(OUTPUT_DIR, "logreg_lvl6_lbfgs_iris_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Artifacts saved to {OUTPUT_DIR}")

def main():
    """Main function to run the softmax regression task."""
    print("=" * 60)
    print("Logistic Regression (LBFGS Iris Softmax)")
    print("=" * 60)

    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    metadata = get_task_metadata()
    print(f"Task: {metadata['task_name']}")

    print("\nCreating dataloaders...")
    train_loader, val_loader, X_train, X_val, y_train, y_val, _ = make_dataloaders(
        batch_size=64, train_ratio=0.8, seed=42, device=device
    )
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    print("\nBuilding model...")
    model = build_model(input_dim=4, num_classes=3, device=device)
    print(f"Model: {model}")

    criterion = nn.CrossEntropyLoss()

    print("\nTraining model with LBFGS...")
    train_lbfgs(model, train_loader, criterion, device, max_iter=200, lr=1.0, verbose=True)

    print("\nEvaluating...")
    train_metrics = evaluate(model, train_loader, criterion, device)
    val_metrics = evaluate(model, val_loader, criterion, device)

    print("\nTrain Metrics:")
    print(f"  Loss: {train_metrics['loss']:.4f}")
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"  F1 Macro: {train_metrics['f1_macro']:.4f}")

    print("\nValidation Metrics:")
    print(f"  Loss: {val_metrics['loss']:.4f}")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  F1 Macro: {val_metrics['f1_macro']:.4f}")
    print(f"  Confusion Matrix: {val_metrics['confusion_matrix']}")

    all_metrics = {
        "metadata": metadata,
        "train": train_metrics,
        "validation": val_metrics,
    }

    print("\nSaving artifacts...")
    save_artifacts(model, all_metrics)

    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)

    acc_ok = val_metrics["accuracy"] > 0.90
    f1_ok = val_metrics["f1_macro"] > 0.90

    print(("✓" if acc_ok else "✗") + f" Val Accuracy > 0.90: {val_metrics['accuracy']:.4f}")
    print(("✓" if f1_ok else "✗") + f" Val Macro-F1 > 0.90: {val_metrics['f1_macro']:.4f}")

    passed = acc_ok and f1_ok
    if passed:
        print("PASS: All quality checks passed!")
        return 0
    else:
        print("FAIL: Some quality checks failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
