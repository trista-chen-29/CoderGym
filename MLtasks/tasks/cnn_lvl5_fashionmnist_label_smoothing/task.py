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
    r"""
    CNN on FashionMNIST (Augmentation + Label Smoothing)

    Cross-entropy with label smoothing:
        L = - \sum_c y_c^{(smooth)} \log p_c

    Classification proxy metrics required by protocol:
        - MSE is computed between predicted class probabilities and one-hot targets
        - R2 is computed on flattened probability and one-hot target vectors
    """
    return {
        "task_id": "cnn_lvl5_fashionmnist_label_smoothing",
        "task_name": "CNN on FashionMNIST (Augmentation + Label Smoothing)",
        "series": "Convolutional Neural Networks",
        "level": 5,
        "dataset": "FashionMNIST",
        "num_classes": 10,
        "input_shape": [1, 28, 28],
        "metrics": ["mse", "r2", "accuracy", "macro_f1"],
        "class_names": [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ],
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
    # Load FashionMNIST with augmentation for training and clean transform for validation.
    data_root = os.path.join(os.path.dirname(__file__), "data")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    base_train_dataset = datasets.FashionMNIST(
        root=data_root,
        train=True,
        download=True,
        transform=None,
    )

    n_total = len(base_train_dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator).tolist()
    train_indices = perm[:n_train]
    val_indices = perm[n_train:]

    train_dataset = datasets.FashionMNIST(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )
    val_dataset = datasets.FashionMNIST(
        root=data_root,
        train=True,
        download=True,
        transform=eval_transform,
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

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


class FashionCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.15),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.20),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model() -> nn.Module:
    # Convolutional neural network for image classification on FashionMNIST.
    meta = get_task_metadata()
    return FashionCNN(num_classes=meta["num_classes"])


def _compute_confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        cm[int(t), int(p)] += 1
    return cm


def _macro_f1_from_confusion_matrix(cm: torch.Tensor) -> float:
    f1s = []
    num_classes = cm.shape[0]
    for c in range(num_classes):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        f1s.append(f1)
    return float(np.mean(f1s))


def _r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
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
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
) -> Dict:
    # Train CNN with label smoothing and track best model based on validation accuracy.
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_history: List[float] = []
    val_loss_history: List[float] = []
    best_val_acc = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            running_loss += loss.item() * bs
            n_samples += bs

        train_epoch_loss = running_loss / max(n_samples, 1)
        loss_history.append(train_epoch_loss)

        val_metrics = evaluate(model, val_loader, device, label_smoothing=label_smoothing)
        val_loss_history.append(val_metrics["cross_entropy"])

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train CE: {train_epoch_loss:.6f} | "
            f"Val CE: {val_metrics['cross_entropy']:.6f} | "
            f"Val Acc: {val_metrics['accuracy']:.6f} | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.6f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "loss_history": loss_history,
        "val_loss_history": val_loss_history,
        "best_val_accuracy": best_val_acc,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_smoothing: float = 0.1,
) -> Dict:
    # Compute classification metrics (accuracy, macro-F1) and
    # regression-style metrics (MSE, R2) required by the protocol.
    model.eval()
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    logits_all = []
    labels_all = []
    running_loss = 0.0
    n_samples = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = criterion(logits, yb)

        bs = xb.size(0)
        running_loss += loss.item() * bs
        n_samples += bs

        logits_all.append(logits.cpu())
        labels_all.append(yb.cpu())

    logits_all = torch.cat(logits_all, dim=0)
    y_true = torch.cat(labels_all, dim=0)

    probs = torch.softmax(logits_all, dim=1)
    y_pred = probs.argmax(dim=1)

    num_classes = probs.shape[1]
    one_hot = torch.nn.functional.one_hot(y_true, num_classes=num_classes).float()

    mse = torch.mean((one_hot - probs) ** 2).item()
    r2 = _r2_score(one_hot, probs)
    accuracy = (y_pred == y_true).float().mean().item()

    cm = _compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    macro_f1 = _macro_f1_from_confusion_matrix(cm)
    class_accuracy = (cm.diag().float() / (cm.sum(dim=1).float() + 1e-8)).tolist()

    return {
        "mse": float(mse),
        "r2": float(r2),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "cross_entropy": float(running_loss / max(n_samples, 1)),
        "per_class_accuracy": [float(x) for x in class_accuracy],
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    X = X.to(device)
    logits = model(X)
    probs = torch.softmax(logits, dim=1)
    return probs.cpu()


def save_artifacts(
    model: nn.Module,
    metrics: Dict,
    history: Dict,
    output_dir: str = None,
) -> None:
    # Save trained model, metrics, training curve, and confusion matrix.
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(output_dir, "fashionmnist_cnn_model.pt"))

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(history["loss_history"], label="train_ce")
    plt.plot(history["val_loss_history"], label="val_ce")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("FashionMNIST CNN Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cnn_lvl5_loss_curve.png"))
    plt.close()

    cm = np.array(metrics["validation"]["confusion_matrix"], dtype=np.int64)
    class_names = get_task_metadata()["class_names"]

    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Validation Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cnn_lvl5_confusion_matrix.png"))
    plt.close()


def _print_metrics(split_name: str, metrics: Dict) -> None:
    print(f"{split_name} metrics:")
    for k, v in metrics.items():
        if k == "confusion_matrix":
            print(f"  {k}: [saved in metrics.json]")
        elif isinstance(v, list):
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
            batch_size=128,
            val_ratio=0.1,
            seed=42,
        )

        model = build_model().to(device)

        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=10,
            lr=1e-3,
            weight_decay=1e-4,
            label_smoothing=0.1,
        )

        train_metrics = evaluate(model, train_loader, device, label_smoothing=0.1)
        val_metrics = evaluate(model, val_loader, device, label_smoothing=0.1)

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

        assert val_metrics["accuracy"] > 0.85, f"Validation accuracy too low: {val_metrics['accuracy']:.6f}"
        assert val_metrics["macro_f1"] > 0.84, f"Validation macro-F1 too low: {val_metrics['macro_f1']:.6f}"
        assert val_metrics["mse"] < 0.08, f"Validation MSE too high: {val_metrics['mse']:.6f}"

        print("\nTask passed all validation thresholds.")
        sys.exit(0)

    except Exception as e:
        print(f"\nTask failed: {e}")
        sys.exit(1)
