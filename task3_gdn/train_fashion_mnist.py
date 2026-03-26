"""
Task 3-2 / 3-3: Train & evaluate GDN classifier on Fashion-MNIST

Usage:
    python train_fashion_mnist.py

Outputs:
    - training_curves.png   (loss & accuracy curves)
    - gdn_fashion_mnist.pth (saved model weights)
"""

import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Ensure the script can import from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gated_deltanet import GDNClassifier


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # Model
    "img_size":      28,
    "patch_size":    4,       # 4×4 patches → 7×7 = 49 tokens
    "hidden_dim":    64,
    "num_heads":     4,
    "num_layers":    3,
    "mlp_expansion": 2,
    "num_classes":   10,
    # Training
    "epochs":        10,
    "batch_size":    128,
    "lr":            1e-3,
    "weight_decay":  1e-4,
}

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════════

FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def get_dataloaders(batch_size: int = 128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    data_dir = os.path.join(SAVE_DIR, "data")

    train_set = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform,
    )
    test_set = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform,
    )

    # num_workers=0 for Windows compatibility
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluate
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


# ═══════════════════════════════════════════════════════════════════════════════
# Train
# ═══════════════════════════════════════════════════════════════════════════════

def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> dict:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = correct = total = 0
        t0 = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(-1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        train_loss = epoch_loss / total
        train_acc  = correct / total
        test_acc   = evaluate(model, test_loader, device)
        elapsed    = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch {epoch:>2d}/{epochs} | "
            f"loss {train_loss:.4f} | "
            f"train_acc {train_acc:.4f} | "
            f"test_acc {test_acc:.4f} | "
            f"lr {scheduler.get_last_lr()[0]:.6f} | "
            f"{elapsed:.1f}s"
        )

    return history


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_history(history: dict, save_path: str):
    """Plot training loss and accuracy curves, save to file."""
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed — skipping plot")
        return

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Loss ──────────────────────────────────────────────────────────────────
    ax1.plot(epochs, history["train_loss"], "o-", markersize=3, label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    ax2.plot(epochs, history["train_acc"],  "o-", markersize=3, label="Train Acc")
    ax2.plot(epochs, history["test_acc"],   "s-", markersize=3, label="Test Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Classification Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {CONFIG}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, test_loader = get_dataloaders(batch_size=CONFIG["batch_size"])
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GDNClassifier(
        img_size=CONFIG["img_size"],
        patch_size=CONFIG["patch_size"],
        hidden_dim=CONFIG["hidden_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        mlp_expansion=CONFIG["mlp_expansion"],
        num_classes=CONFIG["num_classes"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")
    print(model)
    print()

    # ── Train ─────────────────────────────────────────────────────────────────
    history = train(
        model, train_loader, test_loader, device,
        epochs=CONFIG["epochs"],
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    # ── Visualize ─────────────────────────────────────────────────────────────
    plot_path = os.path.join(SAVE_DIR, "training_curves.png")
    plot_history(history, save_path=plot_path)

    # ── Save model ────────────────────────────────────────────────────────────
    ckpt_path = os.path.join(SAVE_DIR, "gdn_fashion_mnist.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")

    # ── Final report ──────────────────────────────────────────────────────────
    final_acc = evaluate(model, test_loader, device)
    print(f"\n{'='*50}")
    print(f"Final Test Accuracy: {final_acc:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
