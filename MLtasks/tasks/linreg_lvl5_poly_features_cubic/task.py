"""
Linear Regression (Poly Features Cubic)

Requirements:
- Data: Synthetic y = 0.5x^3 - 2x^2 + x + noise. 80/20 train/val split.
- Features: manual polynomial expansion up to cubic.
- Model: nn.Linear on expanded features, optimizer from torch.optim.
- Evaluation: evaluate() returns MSE and R2 on VALIDATION split.
- Artifacts: save plot of data + fitted curve to outputs/
- Validation: assert validation R2 > 0.85 (exit non-zero on failure)
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

# Ensure output directory exists
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set seeds for reproducibility
def get_task_metadata():
    """Return task metadata."""
    return {
        "task_name": "linreg_lvl5_poly_features_cubic",
        "task_type": "regression",
        "input_type": "continuous",
        "output_type": "continuous",
        "description": "Polynomial regression with manual cubic features; nn.Linear on expanded features."
    }

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Get device (cuda/cpu)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _poly_features(x: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    Build polynomial features [1, x, x^2, ..., x^degree].
    x is shape (N,) or (N,1). Returns shape (N, degree+1).
    """
    x = np.asarray(x).reshape(-1)
    return np.column_stack([x ** i for i in range(degree + 1)])

def make_dataloaders(
    n_samples=1000,
    train_ratio=0.8,
    batch_size=64,
    noise_std=2.0,
    degree=3,
    x_range=(-3.0, 3.0),
    random_state=42,
    device=None,
):
    """
    Create synthetic dataset:
      y = 0.5x^3 - 2x^2 + x + noise

    Returns:
      train_loader, val_loader, X_train_np, X_val_np, y_train_np, y_val_np
    """
    if device is None:
        device = get_device()

    set_seed(random_state)
    
    # Sample x uniformly
    x = np.random.uniform(x_range[0], x_range[1], size=n_samples)

    # True function (cubic)
    y = 0.5 * (x ** 3) - 2.0 * (x ** 2) + 1.0 * x + np.random.normal(0.0, noise_std, size=n_samples)

    # Build polynomial features up to cubic
    X = _poly_features(x, degree=degree)

    # Train/val split (80/20)
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    n_train = int(train_ratio * n_samples)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    X_train_np, X_val_np = X[train_idx], X[val_idx]
    y_train_np, y_val_np = y[train_idx], y[val_idx]

    # To tensors
    X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train_np, dtype=torch.float32, device=device).unsqueeze(1)
    X_val = torch.tensor(X_val_np, dtype=torch.float32, device=device)
    y_val = torch.tensor(y_val_np, dtype=torch.float32, device=device).unsqueeze(1)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_train_np, X_val_np, y_train_np, y_val_np

class PolynomialRegressionModel(nn.Module):
    """Linear model over polynomial features."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

def build_model(input_dim, device=None):
    """Build the model."""
    if device is None:
        device = get_device()
    
    model = PolynomialRegressionModel(input_dim).to(device)
    return model

def train(model, train_loader, val_loader, device=None, epochs=200, lr=0.01, weight_decay=0.0):
    """
    Train model with torch.optim (SGD/Adam allowed). We'll use Adam for stability.
    Returns training history lists.
    """
    if device is None:
        device = get_device()
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)

        # Validation loss
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(X_batch)
                vloss += criterion(preds, y_batch).item()
            avg_val_loss = vloss / max(1, len(val_loader))

        val_losses.append(avg_val_loss)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses

def evaluate(model, data_loader, device=None):
    """
    Evaluate and return:
      - mse
      - r2
    """
    if device is None:
        device = get_device()
    
    model.eval()
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch)

            preds_all.append(preds.detach().cpu().numpy())
            targets_all.append(y_batch.detach().cpu().numpy())

    y_pred = np.vstack(preds_all).reshape(-1)
    y_true = np.vstack(targets_all).reshape(-1)

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {"mse": float(mse), "r2": float(r2)}

def predict(model, X, device=None):
    """Predict for numpy array X of shape (N, d)."""
    if device is None:
        device = get_device()
    
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        preds = model(X_tensor).detach().cpu().numpy()
    return preds

def save_artifacts(
    model,
    train_losses,
    val_losses,
    X_train_np,
    y_train_np,
    X_val_np,
    y_val_np,
    output_dir=OUTPUT_DIR,
    filename_prefix="linreg_lvl5_poly_features_cubic",
):
    """Save plot + metrics JSON to outputs/."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot: data + fitted curve
    # Build a smooth x grid and predict
    degree = X_train_np.shape[1] - 1
    x_min = min(X_train_np[:, 1].min(), X_val_np[:, 1].min())
    x_max = max(X_train_np[:, 1].max(), X_val_np[:, 1].max())
    x_grid = np.linspace(x_min, x_max, 400)
    X_grid = _poly_features(x_grid, degree=degree)
    y_grid_pred = predict(model, X_grid).reshape(-1)

    plt.figure(figsize=(10, 5))
    plt.scatter(X_train_np[:, 1], y_train_np, s=12, alpha=0.5, label="train")
    plt.scatter(X_val_np[:, 1], y_val_np, s=12, alpha=0.5, label="val")
    plt.plot(x_grid, y_grid_pred, linewidth=2, label="fitted curve")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Cubic Polynomial Regression Fit")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{filename_prefix}_fit.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # Loss curves
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("mse loss")
    plt.title("Training Curves")
    plt.legend()
    plt.tight_layout()
    loss_path = os.path.join(output_dir, f"{filename_prefix}_loss.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()

    # Save model
    model_path = os.path.join(output_dir, f"{filename_prefix}_model.pt")
    torch.save(model.state_dict(), model_path)

    print(f"Artifacts saved to {output_dir}")

def main():
    """Main function to run the task."""
    set_seed(42)

    print("=" * 60)
    print("Linear Regression (Poly Features Cubic)")
    print("=" * 60)

    device = get_device()
    print(f"Using device: {device}")

    print("\nCreating dataloaders...")
    train_loader, val_loader, X_train_np, X_val_np, y_train_np, y_val_np = make_dataloaders(
        n_samples=1000,
        train_ratio=0.8,
        batch_size=64,
        noise_std=2.0,
        degree=3,
        random_state=42,
        device=device,
    )
    print(f"Training samples: {len(y_train_np)}, Validation samples: {len(y_val_np)}")

    print("\nBuilding model...")
    input_dim = X_train_np.shape[1]  # degree+1 = 4 for cubic
    model = build_model(input_dim=input_dim, device=device)
    print(f"Model: {model}")

    print("\nTraining model...")
    train_losses, val_losses = train(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=200,
        lr=0.01,
        weight_decay=0.0,
    )

    print("\nEvaluating model...")
    train_metrics = evaluate(model, train_loader, device=device)
    val_metrics = evaluate(model, val_loader, device=device)

    print("\nMetrics:")
    print(f"  Train - MSE: {train_metrics['mse']:.4f}, R2: {train_metrics['r2']:.4f}")
    print(f"  Val   - MSE: {val_metrics['mse']:.4f}, R2: {val_metrics['r2']:.4f}")

    # Save artifacts + metrics json
    metrics_out = {
        "train": train_metrics,
        "validation": val_metrics,
        "metadata": get_task_metadata(),
        "config": {
            "degree": 3,
            "noise_std": 2.0,
            "epochs": 200,
            "lr": 0.01,
            "batch_size": 64,
        },
    }

    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)

    print("\nSaving artifacts...")
    save_artifacts(
        model,
        train_losses,
        val_losses,
        X_train_np,
        y_train_np,
        X_val_np,
        y_val_np,
        output_dir=OUTPUT_DIR,
        filename_prefix="linreg_lvl5_poly_features_cubic",
    )

    # Quality checks (required)
    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)

    r2_ok = val_metrics["r2"] > 0.85
    print(("✓" if r2_ok else "✗") + f" Val R2 > 0.85: {val_metrics['r2']:.4f}")

    if not r2_ok:
        print("FAIL: Quality checks failed.")
        return 1

    print("PASS: All quality checks passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
