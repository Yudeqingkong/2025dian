"""
Task 1: MLP on Iris Dataset (3-class classification)
- One hidden layer MLP built with nn.Module
- Softmax implemented manually using PyTorch primitives
- Train/test split, accuracy evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Manual Softmax (no nn.Softmax / F.softmax) ──────────────────────────────
def softmax(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable softmax along the last dimension."""
    x_shifted = x - x.max(dim=-1, keepdim=True).values   # stability trick
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)


# ── Model ────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    """Single hidden-layer MLP for 3-class classification.

    Architecture: Linear(4 → 64) → ReLU → Linear(64 → 3)
    The forward pass returns raw logits; softmax is applied separately
    when we need probabilities (not needed by CrossEntropyLoss).
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, num_classes: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits (shape: [batch, num_classes])."""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities using our manual softmax."""
        logits = self.forward(x)
        return softmax(logits)


# ── Data preparation ─────────────────────────────────────────────────────────
def load_data(test_size: float = 0.2, random_state: int = 42):
    iris = load_iris()
    X, y = iris.data, iris.target                         # (150, 4), (150,)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test  = torch.tensor(y_test,  dtype=torch.long)

    return X_train, X_test, y_train, y_test


# ── Training loop ─────────────────────────────────────────────────────────────
def train(model: nn.Module, X_train, y_train, epochs: int = 200, lr: float = 1e-2):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(X_train)          # forward pass
        loss   = criterion(logits, y_train)

        loss.backward()                  # compute gradients
        optimizer.step()                 # update weights

        if epoch % 20 == 0:
            acc = evaluate(model, X_train, y_train)
            print(f"Epoch {epoch:>3d} | loss: {loss.item():.4f} | train acc: {acc:.4f}")


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model: nn.Module, X, y) -> float:
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(X)       # uses our manual softmax
        preds = probs.argmax(dim=-1)
        accuracy = (preds == y).float().mean().item()
    return accuracy


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)

    X_train, X_test, y_train, y_test = load_data()
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    model = MLP(input_dim=4, hidden_dim=64, num_classes=3)
    print(model)

    train(model, X_train, y_train, epochs=200, lr=1e-2)

    test_acc = evaluate(model, X_test, y_test)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    # Quick sanity-check: verify manual softmax probabilities sum to 1
    with torch.no_grad():
        sample_probs = model.predict_proba(X_test[:5])
        print("\nSample probabilities (first 5 test samples):")
        print(sample_probs)
        print("Row sums (should be 1.0):", sample_probs.sum(dim=-1))
