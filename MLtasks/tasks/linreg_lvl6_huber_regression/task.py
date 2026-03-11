"""
Linear Regression (Huber Regression)
Synthetic linear data with ~10% strong outliers. Train/val split 80/20.
Uses nn.Linear + SmoothL1Loss (Huber). Saves scatter + fitted line to outputs/.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Set seeds for reproducibility
def get_task_metadata():
    """Return task metadata."""
    return {
        "task_name": "linreg_lvl6_huber_regression",
        "task_type": "regression",
        "description": "Linear regression with Huber loss on data containing strong outliers."
    }

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Get device (cuda/cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_dataloaders(
    n_samples: int = 1000,
    train_ratio: float = 0.8,
    batch_size: int = 64,
    noise_std: float = 1.0,
    outlier_frac: float = 0.10,
    outlier_scale: float = 20.0,
    seed: int = 42,
    device=None,
):
    """
    Synthetic linear data with outliers:
      y = 2x + 3 + noise
    Then ~10% points get a large additive outlier.
    """
    if device is None:
        device = get_device()
    set_seed(seed)
    
    # x in a reasonable range
    x = np.random.uniform(-5.0, 5.0, size=(n_samples, 1)).astype(np.float32)

    # base linear relation
    y = 2.0 * x + 3.0 + np.random.randn(n_samples, 1).astype(np.float32) * noise_std

    # inject outliers
    n_out = int(outlier_frac * n_samples)
    out_idx = np.random.choice(n_samples, size=n_out, replace=False)
    y[out_idx] += (np.random.randn(n_out, 1).astype(np.float32) * outlier_scale)

    # split
    n_train = int(train_ratio * n_samples)
    perm = np.random.permutation(n_samples)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    X_train, y_train = x[train_idx], y[train_idx]
    X_val, y_val = x[val_idx], y[val_idx]

    # tensors
    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # return numpy too (for plotting)
    return train_loader, val_loader, X_train, y_train, X_val, y_val, out_idx

def build_model(device=None):
    """Build the model."""
    if device is None:
        device = get_device()
    model = nn.Linear(1, 1).to(device)
    return model

def train(
    model,
    train_loader,
    val_loader,
    device=None,
    epochs: int = 300,
    lr: float = 0.05,
    verbose_every: int = 50,
):
    """
    Train with SmoothL1Loss (Huber) to reduce sensitivity to outliers.
    """
    if device is None:
        device = get_device()
    
    criterion = nn.SmoothL1Loss(beta=0.5)  # Huber
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    val_loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            running += loss.item()

        avg_train = running / len(train_loader)
        loss_history.append(float(avg_train))

        # val loss (Huber) for tracking
        model.eval()
        v_running = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                pred = model(Xb.to(device))
                v_running += criterion(pred, yb.to(device)).item()
        avg_val = v_running / len(val_loader)
        val_loss_history.append(float(avg_val))

        if verbose_every and epoch % verbose_every == 0:
            print(f"Epoch [{epoch}/{epochs}], Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")

    return {"loss_history": loss_history, "val_loss_history": val_loss_history}

def evaluate(model, data_loader, device=None):
    """
    MUST return standard metrics on VALIDATION split (and train when called):
      - MSE
      - R2
    """
    if device is None:
        device = get_device()

    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for Xb, yb in data_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            pred = model(Xb)
            preds.append(pred.cpu().numpy())
            targets.append(yb.cpu().numpy())

    preds = np.vstack(preds)
    targets = np.vstack(targets)

    mse = float(mean_squared_error(targets, preds))
    r2 = float(r2_score(targets, preds))

    return {"mse": mse, "r2": r2}

def predict(model, X, device=None):
    """Make predictions."""
    if device is None:
        device = get_device()

    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
        if Xt.ndim == 1:
            Xt = Xt.reshape(-1, 1)
        pred = model(Xt).cpu().numpy()
    return pred

def save_artifacts(
    model,
    train_metrics,
    val_metrics,
    history,
    X_train,
    y_train,
    X_val,
    y_val,
    output_dir=None,
):
    """Save model artifacts and visualization."""
    # outputs/ next to this file (repo style you used in softmax)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if output_dir is None:
        output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # save metrics json
    metrics_path = os.path.join(output_dir, "linreg_lvl6_huber_metrics.json")
    payload = {
        "metadata": get_task_metadata(),
        "train": train_metrics,
        "validation": val_metrics,
        "history": history,
        "learned_params": {
            "weight": float(model.weight.detach().cpu().numpy().reshape(-1)[0]),
            "bias": float(model.bias.detach().cpu().numpy().reshape(-1)[0]),
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)

    # save model
    model_path = os.path.join(output_dir, "linreg_lvl6_huber_model.pt")
    torch.save(model.state_dict(), model_path)

    # scatter plot showing outliers + fitted line
    plot_path = os.path.join(output_dir, "linreg_lvl6_huber_fit.png")
    plot_outliers_and_fit(model, X_train, y_train, X_val, y_val, plot_path)

    print(f"Artifacts saved to {output_dir}")

def plot_outliers_and_fit(model, X_train, y_train, X_val, y_val, save_path):
    """Create and save visualization of the fit."""
    # Build a smooth x grid for line
    x_all = np.vstack([X_train, X_val])
    x_min, x_max = float(x_all.min()) - 1.0, float(x_all.max()) + 1.0
    x_line = np.linspace(x_min, x_max, 300).reshape(-1, 1).astype(np.float32)
    y_line = predict(model, x_line)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, s=18, alpha=0.6, label="Train (with outliers)")
    plt.scatter(X_val, y_val, s=24, alpha=0.8, label="Val")
    plt.plot(x_line, y_line, linewidth=2, label="Fitted line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Huber Regression: Outliers + Fitted Line")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def main():
    """Main function to run the task."""
    print("=" * 60)
    print("Linear Regression (Huber Regression)")
    print("=" * 60)

    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    print("\nCreating dataloaders...")
    train_loader, val_loader, X_train, y_train, X_val, y_val, _ = make_dataloaders(
        n_samples=1000,
        train_ratio=0.8,
        batch_size=64,
        noise_std=0.6,
        outlier_frac=0.10,
        outlier_scale=10.0,
        seed=42,
        device=device,
    )
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    print("\nBuilding model...")
    model = build_model(device=device)
    print(f"Model: {model}")

    print("\nTraining model...")
    history = train(model, train_loader, val_loader, device=device, epochs=300, lr=0.05, verbose_every=50)

    print("\nEvaluating model...")
    train_metrics = evaluate(model, train_loader, device=device)
    val_metrics = evaluate(model, val_loader, device=device)

    print("\nMetrics:")
    print(f"  Train - MSE: {train_metrics['mse']:.4f}, R2: {train_metrics['r2']:.4f}")
    print(f"  Val   - MSE: {val_metrics['mse']:.4f}, R2: {val_metrics['r2']:.4f}")

    print("\nSaving artifacts...")
    save_artifacts(model, train_metrics, val_metrics, history, X_train, y_train, X_val, y_val)

    # Quality checks per your JSON
    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)

    passed = True

    r2_ok = val_metrics["r2"] > 0.80
    print(("✓" if r2_ok else "✗") + f" Val R2 > 0.80: {val_metrics['r2']:.4f}")
    passed = passed and r2_ok

    # "reasonable threshold" — outliers can inflate MSE, so don't set too tight
    mse_thresh = 120.0
    mse_ok = val_metrics["mse"] < mse_thresh
    print(("✓" if mse_ok else "✗") + f" Val MSE < {mse_thresh}: {val_metrics['mse']:.4f}")
    passed = passed and mse_ok

    if passed:
        print("PASS: All quality checks passed!")
        return 0
    else:
        print("FAIL: Some quality checks failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
