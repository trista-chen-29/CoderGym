import json
import os
import random
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def get_task_metadata() -> Dict:
    """
    MLP (Tabular Multilabel Classification)

    This task trains a multilabel MLP classifier on a synthetic tabular dataset.

    Sigmoid:
        \sigma(z) = \frac{1}{1 + e^{-z}}

    Binary cross-entropy with logits:
        BCE(y, z) = - \frac{1}{N} \sum_i \left[y_i \log(\sigma(z_i)) + (1-y_i)\log(1-\sigma(z_i))\right]
    """
    return {
        "task_id": "mlp_lvl5_tabular_multilabel",
        "task_name": "MLP (Tabular Multilabel Classification)",
        "series": "Neural Networks (MLP)",
        "level": 5,
        "n_features": 20,
        "n_labels": 4,
        "metrics": ["mse", "r2", "subset_accuracy", "micro_f1", "macro_f1"],
    }


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _generate_multilabel_data(
    n_samples: int = 2200,
    n_features: int = 20,
    n_labels: int = 4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    # Create correlated features and nonlinear label dependencies
    # so the task is non-trivial.
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n_samples, n_features)).astype(np.float32)

    W = rng.normal(0.0, 1.0, size=(n_features, n_labels)).astype(np.float32)
    b = rng.normal(0.0, 0.35, size=(n_labels,)).astype(np.float32)

    X[:, 5:10] += 0.5 * X[:, 0:5]
    X[:, 10:15] -= 0.3 * X[:, 0:5]

    logits = X @ W + b
    logits[:, 1] += 0.7 * X[:, 0] - 0.4 * X[:, 3]
    logits[:, 2] += 0.8 * X[:, 7] + 0.5 * X[:, 2]
    logits[:, 3] += 0.6 * X[:, 1] - 0.6 * X[:, 6]

    probs = 1.0 / (1.0 + np.exp(-logits))

    # Convert probabilities to multilabel targets using fixed thresholds.
    # Ensure at least one label per sample.
    thresholds = np.array([0.48, 0.52, 0.50, 0.54], dtype=np.float32)
    y = (probs > thresholds).astype(np.float32)

    empty_rows = np.where(y.sum(axis=1) == 0)[0]
    if len(empty_rows) > 0:
        top_label = np.argmax(probs[empty_rows], axis=1)
        y[empty_rows] = 0.0
        y[empty_rows, top_label] = 1.0

    return X.astype(np.float32), y.astype(np.float32)


def make_dataloaders(
    batch_size: int = 64,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict]:
    # Generate synthetic multilabel dataset and standardize features using training stats.
    meta = get_task_metadata()
    X, y = _generate_multilabel_data(
        n_samples=2200,
        n_features=meta["n_features"],
        n_labels=meta["n_labels"],
        seed=seed,
    )

    split_idx = int(len(X) * train_ratio)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    data_meta = {
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "feature_mean_shape": list(mean.shape),
        "feature_std_shape": list(std.shape),
        "label_prevalence_train": y_train.mean(axis=0).tolist(),
        "label_prevalence_val": y_val.mean(axis=0).tolist(),
    }

    return train_loader, val_loader, data_meta


class MultilabelMLP(nn.Module):
    def __init__(self, input_dim: int = 20, output_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model() -> nn.Module:
    # Simple feedforward MLP for multilabel classification.
    meta = get_task_metadata()
    return MultilabelMLP(input_dim=meta["n_features"], output_dim=meta["n_labels"])


def _binary_f1(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return float(f1)


def _macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    f1s = []
    for j in range(y_true.shape[1]):
        f1s.append(_binary_f1(y_true[:, j], y_pred[:, j]))
    return float(np.mean(f1s))


def _micro_f1(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return float(f1)


def _subset_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    exact_match = (y_true == y_pred).all(dim=1).float().mean().item()
    return float(exact_match)


def _r2_score(y_true: torch.Tensor, y_pred_prob: torch.Tensor) -> float:
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred_prob.view(-1)
    ss_res = torch.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = torch.sum((y_true_flat - torch.mean(y_true_flat)) ** 2)
    if ss_tot.item() < 1e-12:
        return 0.0
    return float(1.0 - (ss_res / ss_tot).item())


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 35,
    lr: float = 1e-3,
) -> Dict:
    # Train using BCEWithLogitsLoss for multilabel outputs and track best model.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history: List[float] = []
    val_loss_history: List[float] = []
    best_val_micro_f1 = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            batch_size = xb.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

        train_epoch_loss = running_loss / max(n_samples, 1)
        loss_history.append(train_epoch_loss)

        val_metrics = evaluate(model, val_loader, device)
        val_loss_history.append(val_metrics["bce_loss"])

        if val_metrics["micro_f1"] > best_val_micro_f1:
            best_val_micro_f1 = val_metrics["micro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train BCE: {train_epoch_loss:.6f} | "
            f"Val BCE: {val_metrics['bce_loss']:.6f} | "
            f"Val Micro-F1: {val_metrics['micro_f1']:.6f} | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.6f} | "
            f"Val Subset Acc: {val_metrics['subset_accuracy']:.6f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "loss_history": loss_history,
        "val_loss_history": val_loss_history,
        "best_val_micro_f1": best_val_micro_f1,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    # Compute multilabel metrics (F1, subset accuracy) and
    # regression-style metrics (MSE, R2) required by the protocol.
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    logits_all = []
    targets_all = []
    running_loss = 0.0
    n_samples = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        batch_size = xb.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        logits_all.append(logits.cpu())
        targets_all.append(yb.cpu())

    logits_all = torch.cat(logits_all, dim=0)
    y_true = torch.cat(targets_all, dim=0)
    y_prob = torch.sigmoid(logits_all)
    y_pred = (y_prob >= 0.5).float()

    mse = torch.mean((y_true - y_prob) ** 2).item()
    r2 = _r2_score(y_true, y_prob)
    subset_acc = _subset_accuracy(y_true, y_pred)
    micro_f1 = _micro_f1(y_true, y_pred)
    macro_f1 = _macro_f1(y_true, y_pred)
    bce_loss = running_loss / max(n_samples, 1)

    per_label_accuracy = (y_true == y_pred).float().mean(dim=0).tolist()

    return {
        "mse": float(mse),
        "r2": float(r2),
        "subset_accuracy": float(subset_acc),
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "bce_loss": float(bce_loss),
        "per_label_accuracy": [float(v) for v in per_label_accuracy],
    }


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    X = X.to(device)
    logits = model(X)
    probs = torch.sigmoid(logits)
    return probs.cpu()


def save_artifacts(
    model: nn.Module,
    metrics: Dict,
    history: Dict,
    output_dir: str = None,
) -> None:
    # Save trained model, metrics, and training curves.
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(output_dir, "mlp_multilabel_model.pt"))

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(history["loss_history"], label="train_bce")
    plt.plot(history["val_loss_history"], label="val_bce")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLP Multilabel Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mlp_lvl5_loss_curve.png"))
    plt.close()


def _print_metrics(split_name: str, metrics: Dict) -> None:
    print(f"{split_name} metrics:")
    for k, v in metrics.items():
        if isinstance(v, list):
            print(f"  {k}: {[round(float(x), 6) for x in v]}")
        else:
            print(f"  {k}: {float(v):.6f}")


if __name__ == "__main__":
    # Required by pytorch_task_v1:
    # train, evaluate, print metrics, assert thresholds, exit with status code.
    try:
        set_seed(42)
        device = get_device()
        metadata = get_task_metadata()

        print("Task metadata:")
        print(json.dumps(metadata, indent=2))
        print(f"Using device: {device}")

        train_loader, val_loader, data_meta = make_dataloaders(
            batch_size=64,
            train_ratio=0.8,
            seed=42,
        )

        model = build_model().to(device)

        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=35,
            lr=1e-3,
        )

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        print("\nFinal evaluation:")
        _print_metrics("Train", train_metrics)
        _print_metrics("Validation", val_metrics)

        final_metrics = {
            "train": train_metrics,
            "validation": val_metrics,
            "data_meta": data_meta,
        }

        save_artifacts(
            model=model,
            metrics=final_metrics,
            history=history,
        )

        assert val_metrics["micro_f1"] > 0.85, f"Validation micro-F1 too low: {val_metrics['micro_f1']:.6f}"
        assert val_metrics["macro_f1"] > 0.80, f"Validation macro-F1 too low: {val_metrics['macro_f1']:.6f}"
        assert val_metrics["mse"] < 0.18, f"Validation MSE too high: {val_metrics['mse']:.6f}"

        print("\nTask passed all validation thresholds.")
        sys.exit(0)

    except Exception as e:
        print(f"\nTask failed: {e}")
        sys.exit(1)
