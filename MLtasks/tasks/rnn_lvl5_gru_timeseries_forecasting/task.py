import json
import math
import os
import random
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


def get_task_metadata() -> Dict:
    """
    GRU Time-Series Forecasting

    Sequence-to-one regression task using a GRU to predict the next value
    in a synthetic time series with trend, seasonality, and noise.

    Objective:
        MSE = (1/N) * sum_i (y_i - yhat_i)^2
    """
    return {
        "task_id": "rnn_lvl5_gru_timeseries_forecasting",
        "task_name": "GRU Time-Series Forecasting",
        "series": "Sequence Models (RNN/LSTM)",
        "level": 5,
        "input_window": 24,
        "prediction_horizon": 1,
        "metrics": ["mse", "r2", "mae"],
    }


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _generate_time_series(n_points: int = 1600, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n_points, dtype=np.float32)

    trend = 0.003 * t
    season1 = 0.8 * np.sin(2 * np.pi * t / 24.0)
    season2 = 0.35 * np.sin(2 * np.pi * t / 7.0)
    noise = rng.normal(0.0, 0.12, size=n_points).astype(np.float32)

    series = trend + season1 + season2 + noise
    return series.astype(np.float32)


def _make_windows(series: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(len(series) - window_size):
        xs.append(series[i:i + window_size])
        ys.append(series[i + window_size])
    X = np.array(xs, dtype=np.float32)
    y = np.array(ys, dtype=np.float32)
    return X, y



def make_dataloaders(
    batch_size: int = 64,
    window_size: int = 24,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict]:
    # Generate synthetic time series and create sliding window sequences.
    # Normalize using training statistics only (avoid data leakage).
    series = _generate_time_series(n_points=1600, seed=seed)
    X, y = _make_windows(series, window_size)

    split_idx = int(len(X) * train_ratio)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    train_mean = X_train.mean()
    train_std = X_train.std() + 1e-8

    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std
    y_train = (y_train - train_mean) / train_std
    y_val = (y_val - train_mean) / train_std

    X_train_t = torch.tensor(X_train).unsqueeze(-1)
    X_val_t = torch.tensor(X_val).unsqueeze(-1)
    y_train_t = torch.tensor(y_train).unsqueeze(-1)
    y_val_t = torch.tensor(y_val).unsqueeze(-1)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    meta = {
        "train_mean": float(train_mean),
        "train_std": float(train_std),
        "window_size": window_size,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "raw_series": series.tolist(),
    }
    return train_loader, val_loader, meta


class GRUForecaster(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]
        pred = self.head(last_hidden)
        return pred


def build_model() -> nn.Module:
    # GRU-based model for sequence-to-one time series forecasting.
    return GRUForecaster(input_size=1, hidden_size=48, num_layers=1, dropout=0.0)


def _r2_score_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    if ss_tot.item() < 1e-12:
        return 0.0
    return float(1.0 - (ss_res / ss_tot).item())


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 40,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    use_scheduler: bool = True,
) -> Dict:
    # Train model and track best checkpoint based on validation MSE.
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    loss_history: List[float] = []
    val_loss_history: List[float] = []
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            batch_size = xb.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

        train_epoch_loss = running_loss / max(n_samples, 1)
        loss_history.append(train_epoch_loss)

        val_metrics = evaluate(model, val_loader, device)
        val_loss_history.append(val_metrics["mse"])

        if val_metrics["mse"] < best_val_loss:
            best_val_loss = val_metrics["mse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train MSE: {train_epoch_loss:.6f} | "
            f"Val MSE: {val_metrics['mse']:.6f} | "
            f"Val R2: {val_metrics['r2']:.6f} | "
            f"Val MAE: {val_metrics['mae']:.6f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "loss_history": loss_history,
        "val_loss_history": val_loss_history,
        "best_val_mse": best_val_loss,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    # Compute required metrics (MSE, R2, MAE) on validation data.
    model.eval()

    preds_all = []
    targets_all = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        preds = model(xb)
        preds_all.append(preds.cpu())
        targets_all.append(yb.cpu())

    y_pred = torch.cat(preds_all, dim=0)
    y_true = torch.cat(targets_all, dim=0)

    mse = torch.mean((y_true - y_pred) ** 2).item()
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    r2 = _r2_score_torch(y_true, y_pred)

    return {
        "mse": float(mse),
        "r2": float(r2),
        "mae": float(mae),
    }


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    X = X.to(device)
    preds = model(X)
    return preds.cpu()


def save_artifacts(
    model: nn.Module,
    metrics: Dict,
    history: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: str = None,
) -> None:
    # Save model, metrics, and plots for verification and analysis.
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "gru_timeseries_model.pt")
    torch.save(model.state_dict(), model_path)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    history_path = os.path.join(output_dir, "history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(history["loss_history"], label="train_mse")
    plt.plot(history["val_loss_history"], label="val_mse")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GRU Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rnn_lvl5_loss_curve.png"))
    plt.close()

    model.eval()
    preds_all = []
    targets_all = []

    for xb, yb in val_loader:
        xb = xb.to(device)
        preds = model(xb).cpu()
        preds_all.append(preds)
        targets_all.append(yb)

    y_pred = torch.cat(preds_all, dim=0).view(-1).detach().numpy()
    y_true = torch.cat(targets_all, dim=0).view(-1).detach().numpy()

    plt.figure(figsize=(10, 5))
    n_plot = min(200, len(y_true))
    plt.plot(y_true[:n_plot], label="target")
    plt.plot(y_pred[:n_plot], label="prediction")
    plt.xlabel("Validation Time Step")
    plt.ylabel("Normalized Value")
    plt.title("Validation Prediction vs Target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rnn_lvl5_prediction_vs_target.png"))
    plt.close()


def _print_metrics(split_name: str, metrics: Dict) -> None:
    print(f"{split_name} metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    # Required by pytorch_task_v1:
    # train, evaluate (train + val), print metrics, assert thresholds, exit with status code.
    try:
        set_seed(42)
        device = get_device()
        metadata = get_task_metadata()

        print("Task metadata:")
        print(json.dumps(metadata, indent=2))
        print(f"Using device: {device}")

        train_loader, val_loader, data_meta = make_dataloaders(
            batch_size=64,
            window_size=metadata["input_window"],
            train_ratio=0.8,
            seed=42,
        )

        model = build_model().to(device)

        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=40,
            lr=1e-3,
            grad_clip=1.0,
            use_scheduler=True,
        )

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        print("\nFinal evaluation:")
        _print_metrics("Train", train_metrics)
        _print_metrics("Validation", val_metrics)

        final_metrics = {
            "train": train_metrics,
            "validation": val_metrics,
            "data_meta": {
                "n_train": data_meta["n_train"],
                "n_val": data_meta["n_val"],
                "window_size": data_meta["window_size"],
            },
        }

        save_artifacts(
            model=model,
            metrics=final_metrics,
            history=history,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )

        assert val_metrics["r2"] > 0.85, f"Validation R2 too low: {val_metrics['r2']:.6f}"
        assert val_metrics["mae"] < 0.45, f"Validation MAE too high: {val_metrics['mae']:.6f}"
        assert val_metrics["mse"] < 0.30, f"Validation MSE too high: {val_metrics['mse']:.6f}"

        print("\nTask passed all validation thresholds.")
        sys.exit(0)

    except Exception as e:
        print(f"\nTask failed: {e}")
        sys.exit(1)
