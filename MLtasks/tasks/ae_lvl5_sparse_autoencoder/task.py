import json
import os
import random
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_task_metadata() -> Dict:
    """
    Sparse Autoencoder

    Reconstruction objective:
        MSE = (1/N) * sum_i ||x_i - xhat_i||^2

    Sparse objective:
        L = ReconstructionLoss + lambda * mean(|z|)

    where z is the latent activation.
    """
    return {
        "task_id": "ae_lvl5_sparse_autoencoder",
        "task_name": "Sparse Autoencoder",
        "series": "Autoencoders",
        "level": 5,
        "dataset": "MNIST",
        "input_dim": 28 * 28,
        "latent_dim": 32,
        "metrics": ["mse", "r2", "latent_mean_abs", "latent_fraction_near_zero"],
    }


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(
    batch_size: int = 128,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict]:
    # Load MNIST and split into training and validation sets.
    data_root = os.path.join(os.path.dirname(__file__), "data")
    transform = transforms.ToTensor()

    full_train = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )

    n_total = len(full_train)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator).tolist()
    train_indices = perm[:n_train]
    val_indices = perm[n_train:]

    train_subset = Subset(full_train, train_indices)
    val_subset = Subset(full_train, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    data_meta = {
        "n_train": len(train_subset),
        "n_val": len(val_subset),
        "batch_size": batch_size,
        "val_ratio": val_ratio,
    }
    return train_loader, val_loader, data_meta


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int = 784, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def build_model() -> nn.Module:
    # Fully connected autoencoder with a low-dimensional latent representation.
    meta = get_task_metadata()
    return Autoencoder(input_dim=meta["input_dim"], latent_dim=meta["latent_dim"])


def _flatten_batch(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)


def _r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    if ss_tot.item() < 1e-12:
        return 0.0
    return float(1.0 - (ss_res / ss_tot).item())


def _latent_stats(z: torch.Tensor, near_zero_threshold: float = 0.05) -> Dict:
    mean_abs = torch.mean(torch.abs(z)).item()
    frac_near_zero = (torch.abs(z) < near_zero_threshold).float().mean().item()
    return {
        "latent_mean_abs": float(mean_abs),
        "latent_fraction_near_zero": float(frac_near_zero),
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 12,
    lr: float = 1e-3,
    l1_lambda: float = 1e-3,
) -> Dict:
    # Train sparse autoencoder using reconstruction loss + L1 penalty on latent activations.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    recon_criterion = nn.MSELoss()

    loss_history: List[float] = []
    val_loss_history: List[float] = []
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_samples = 0

        for xb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            xb = _flatten_batch(xb)

            optimizer.zero_grad()
            recon, z = model(xb)

            recon_loss = recon_criterion(recon, xb)
            sparse_penalty = torch.mean(torch.abs(z))
            loss = recon_loss + l1_lambda * sparse_penalty

            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            running_loss += loss.item() * bs
            n_samples += bs

        train_epoch_loss = running_loss / max(n_samples, 1)
        loss_history.append(train_epoch_loss)

        val_metrics = evaluate(model, val_loader, device)
        val_loss_history.append(val_metrics["mse"])

        if val_metrics["mse"] < best_val_loss:
            best_val_loss = val_metrics["mse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Total Loss: {train_epoch_loss:.6f} | "
            f"Val MSE: {val_metrics['mse']:.6f} | "
            f"Val R2: {val_metrics['r2']:.6f} | "
            f"Val |z| mean: {val_metrics['latent_mean_abs']:.6f} | "
            f"Val frac near zero: {val_metrics['latent_fraction_near_zero']:.6f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "loss_history": loss_history,
        "val_loss_history": val_loss_history,
        "best_val_mse": best_val_loss,
        "l1_lambda": l1_lambda,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    # Compute reconstruction metrics (MSE, R2) and sparsity statistics of latent space.
    model.eval()

    x_true_all = []
    x_recon_all = []
    z_all = []

    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        xb = _flatten_batch(xb)

        recon, z = model(xb)

        x_true_all.append(xb.cpu())
        x_recon_all.append(recon.cpu())
        z_all.append(z.cpu())

    x_true = torch.cat(x_true_all, dim=0)
    x_recon = torch.cat(x_recon_all, dim=0)
    z = torch.cat(z_all, dim=0)

    mse = torch.mean((x_true - x_recon) ** 2).item()
    r2 = _r2_score(x_true, x_recon)
    stats = _latent_stats(z)

    return {
        "mse": float(mse),
        "r2": float(r2),
        "latent_mean_abs": stats["latent_mean_abs"],
        "latent_fraction_near_zero": stats["latent_fraction_near_zero"],
    }


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    X = X.to(device)
    X = _flatten_batch(X)
    recon, _ = model(X)
    return recon.cpu()


def _compute_baseline_latent_stats(
    loader: DataLoader,
    device: torch.device,
    epochs: int = 4,
    lr: float = 1e-3,
) -> Dict:
    # Train baseline autoencoder (no sparsity) for comparison.
    baseline = Autoencoder(input_dim=784, latent_dim=32).to(device)
    optimizer = torch.optim.Adam(baseline.parameters(), lr=lr)
    criterion = nn.MSELoss()

    baseline.train()
    for _ in range(epochs):
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            xb = _flatten_batch(xb)

            optimizer.zero_grad()
            recon, z = baseline(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()

    baseline.eval()
    z_all = []
    mse_values = []

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            xb = _flatten_batch(xb)
            recon, z = baseline(xb)
            z_all.append(z.cpu())
            mse_values.append(torch.mean((xb.cpu() - recon.cpu()) ** 2).item())

    z_all = torch.cat(z_all, dim=0)
    stats = _latent_stats(z_all)
    return {
        "baseline_mse": float(np.mean(mse_values)),
        "baseline_latent_mean_abs": stats["latent_mean_abs"],
        "baseline_latent_fraction_near_zero": stats["latent_fraction_near_zero"],
    }


def save_artifacts(
    model: nn.Module,
    metrics: Dict,
    history: Dict,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: str = None,
) -> None:
    # Save model, metrics, loss curve, and reconstruction examples.
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(output_dir, "sparse_autoencoder_model.pt"))

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(history["loss_history"], label="train_total_loss")
    plt.plot(history["val_loss_history"], label="val_mse")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Sparse Autoencoder Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ae_lvl5_loss_curve.png"))
    plt.close()

    model.eval()
    xb, _ = next(iter(val_loader))
    xb = xb[:8].to(device)
    xb_flat = _flatten_batch(xb)
    recon, _ = model(xb_flat)

    originals = xb.detach().cpu().view(-1, 28, 28).numpy()
    reconstructions = recon.detach().cpu().view(-1, 28, 28).numpy()

    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    for i in range(8):
        axes[0, i].imshow(originals[i], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructions[i], cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_title("Original", fontsize=10)
    axes[1, 0].set_title("Recon", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ae_lvl5_reconstructions.png"))
    plt.close()


def _print_metrics(split_name: str, metrics: Dict) -> None:
    print(f"{split_name} metrics:")
    for k, v in metrics.items():
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
            batch_size=128,
            val_ratio=0.1,
            seed=42,
        )

        baseline_stats = _compute_baseline_latent_stats(
            loader=val_loader,
            device=device,
            epochs=4,
            lr=1e-3,
        )

        print("\nBaseline non-sparse reference:")
        for k, v in baseline_stats.items():
            print(f"  {k}: {v:.6f}")

        model = build_model().to(device)

        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=12,
            lr=1e-3,
            l1_lambda=1e-3,
        )

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        print("\nFinal evaluation:")
        _print_metrics("Train", train_metrics)
        _print_metrics("Validation", val_metrics)

        final_metrics = {
            "train": train_metrics,
            "validation": val_metrics,
            "baseline_reference": baseline_stats,
            "data_meta": data_meta,
        }

        save_artifacts(
            model=model,
            metrics=final_metrics,
            history=history,
            val_loader=val_loader,
            device=device,
        )

        assert val_metrics["mse"] < 0.035, f"Validation MSE too high: {val_metrics['mse']:.6f}"
        assert val_metrics["r2"] > 0.55, f"Validation R2 too low: {val_metrics['r2']:.6f}"
        assert (
            val_metrics["latent_mean_abs"] < baseline_stats["baseline_latent_mean_abs"]
        ), "Sparse autoencoder did not reduce mean latent activation versus baseline."

        print("\nTask passed all validation thresholds.")
        sys.exit(0)

    except Exception as e:
        print(f"\nTask failed: {e}")
        sys.exit(1)
